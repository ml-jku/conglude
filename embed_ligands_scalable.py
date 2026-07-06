import torch
import yaml
import os
import re
import bz2
import gzip
import gc
import time
import shutil
import numpy as np
import argparse
import psutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from conglude.utils.common import read_list_from_txt, write_json, read_json
from conglude.utils.data_processing import LigandProcessor
from conglude.modules.mlp import MLPEncoder


def process_shard_features(
    shard_idx,
    smiles_file,
    output_dir,
    ecfp_radius,
    fp_length,
    calc_descriptors,
    num_workers,
    smiles_batch_size,
    scaler_dir,
    overwrite,
):
    """
    Compute fingerprints and descriptors for a single shard.

    Saves .dat.gz files into <output_dir>/processed/ligand_embeddings/ with
    shard index suffixes. Designed to run in a subprocess.

    Parameters
    ----------
    shard_idx : int
        Index of the shard being processed.
    smiles_file : str
        Path to the shard's SMILES text file.
    output_dir : str
        Base output directory.
    ecfp_radius : int
        Radius for ECFP fingerprint generation.
    fp_length : int
        Length of the fingerprint bit vector.
    calc_descriptors : bool
        Whether to compute molecular descriptors.
    num_workers : int
        Number of parallel workers for RDKit computation.
    smiles_batch_size : int
        Number of SMILES per parallel batch.
    scaler_dir : str
        Directory containing pre-fitted RobustScaler.
    overwrite : bool
        Whether to recompute existing features.

    Returns
    -------
    tuple of (int, bool)
        Shard index and success status.
    """
    emb_dir = os.path.join(output_dir, "processed", "ligand_embeddings")
    fp_name = f"ecfp{2 * ecfp_radius}_{fp_length}"

    fp_gz_path = os.path.join(emb_dir, f"{fp_name}_{shard_idx}.dat.gz")
    desc_gz_path = os.path.join(emb_dir, f"descriptors_{shard_idx}.dat.gz")
    metadata_path = os.path.join(emb_dir, f"metadata_{fp_name}_{shard_idx}.json")

    if not overwrite and os.path.exists(metadata_path):
        fp_exists = os.path.exists(fp_gz_path) or os.path.exists(fp_gz_path.replace(".gz", ""))
        desc_exists = not calc_descriptors or os.path.exists(desc_gz_path) or os.path.exists(desc_gz_path.replace(".gz", ""))
        if fp_exists and desc_exists:
            print(f"[Shard {shard_idx}] Features already exist, skipping.")
            return shard_idx, True

    os.makedirs(emb_dir, exist_ok=True)

    smiles_list = read_list_from_txt(smiles_file)

    index2smiles_path = os.path.join(emb_dir, f"index2smiles_{shard_idx}.json")
    if not os.path.exists(index2smiles_path):
        index2smiles_dict = {str(i): s for i, s in enumerate(smiles_list)}
        write_json(index2smiles_path, index2smiles_dict)

    print(f"[Shard {shard_idx}] Computing features for {len(smiles_list)} SMILES...")

    tmp_dir = os.path.join(output_dir, f"_tmp_shard_{shard_idx}")
    tmp_emb_dir = os.path.join(tmp_dir, "processed", "ligand_embeddings")
    os.makedirs(tmp_emb_dir, exist_ok=True)

    write_json(os.path.join(tmp_emb_dir, "index2smiles.json"),
               {str(i): s for i, s in enumerate(smiles_list)})

    ligand_processor = LigandProcessor(
        dataset_dir=tmp_dir,
        ecfp_radius=ecfp_radius,
        fp_length=fp_length,
        calc_descriptors=calc_descriptors,
        num_workers=num_workers,
        smiles_batch_size=smiles_batch_size,
        scaler_dir=scaler_dir,
        load_scaler=True,
        save_scaler=False,
        save_pt=False,
        show_progress=True,
    )
    ligand_processor.process(smiles_list)

    fp_dat = os.path.join(tmp_emb_dir, f"{fp_name}.dat")
    if os.path.exists(fp_dat):
        target = os.path.join(emb_dir, f"{fp_name}_{shard_idx}.dat")
        shutil.move(fp_dat, target)
        _gzip_and_remove(target)

    if calc_descriptors:
        desc_dat = os.path.join(tmp_emb_dir, "descriptors.dat")
        if os.path.exists(desc_dat):
            target = os.path.join(emb_dir, f"descriptors_{shard_idx}.dat")
            shutil.move(desc_dat, target)
            _gzip_and_remove(target)

    tmp_metadata = os.path.join(tmp_emb_dir, f"metadata_{fp_name}.json")
    if os.path.exists(tmp_metadata):
        shutil.move(tmp_metadata, metadata_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[Shard {shard_idx}] Features complete.")
    return shard_idx, True


def _gzip_and_remove(filepath):
    """Gzip a file in place and remove the original."""
    gz_path = filepath + ".gz"
    with open(filepath, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            while True:
                chunk = f_in.read(64 * 1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
    os.remove(filepath)


def split_input_file(input_file, smiles_dir, shard_size, smiles_column, delimiter, skip_header, overwrite):
    """
    Split a raw SMILES file into numbered shard files.

    Parameters
    ----------
    input_file : str
        Path to the input file (.bz2/.gz/.tsv/.txt).
    smiles_dir : str
        Output directory for shard files.
    shard_size : int
        Number of SMILES per shard.
    smiles_column : int
        Column index (0-based) of SMILES in the input file.
    delimiter : str
        Field delimiter in input file.
    skip_header : bool
        Whether to skip the first line.
    overwrite : bool
        Whether to overwrite existing shard files.

    Returns
    -------
    int
        Number of shards created.
    """
    os.makedirs(smiles_dir, exist_ok=True)

    if not overwrite:
        existing = [f for f in os.listdir(smiles_dir) if re.match(r"smiles_\d+\.txt$", f)]
        if existing:
            print(f"Found {len(existing)} existing shard files in {smiles_dir}, skipping split.")
            return len(existing)

    if input_file.endswith(".bz2"):
        fh = bz2.open(input_file, "rt")
    elif input_file.endswith(".gz"):
        fh = gzip.open(input_file, "rt")
    else:
        fh = open(input_file, "r")

    shard_idx = 0
    line_count = 0
    out_f = None
    start_time = time.time()

    with fh:
        if skip_header:
            fh.readline()

        for line in fh:
            if line_count % shard_size == 0:
                if out_f:
                    out_f.close()
                shard_path = os.path.join(smiles_dir, f"smiles_{shard_idx}.txt")
                out_f = open(shard_path, "w")
                if shard_idx > 0 and shard_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = line_count / elapsed
                    print(f"  Splitting: {line_count:,} lines, {shard_idx} shards, "
                          f"{rate:.0f} lines/s, elapsed {elapsed/3600:.2f}h")
                shard_idx += 1

            smiles = line.split(delimiter, smiles_column + 1)[smiles_column].strip()
            out_f.write(smiles + "\n")
            line_count += 1

        if out_f:
            out_f.close()

    elapsed = time.time() - start_time
    print(f"Split {line_count:,} SMILES into {shard_idx} shards of up to {shard_size:,} "
          f"({elapsed/3600:.2f}h)")
    return shard_idx


def _get_available_memory_gb():
    """Return available system memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def _load_features_from_gz(gz_path, num_ligands, dim):
    """Load a gzipped .dat file into a numpy array."""
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.float32).reshape(num_ligands, dim)


class ShardedLigandEmbedder:
    """
    Scalable ligand embedding pipeline for large SMILES datasets.

    Splits a large SMILES file into shards, computes molecular features
    in parallel, and encodes them using a trained ligand encoder.

    Parameters
    ----------
    checkpoint_path : str
        Directory containing the trained ligand encoder weights.
    smiles_dir : str
        Directory containing smiles_N.txt shard files.
    output_dir : str
        Base output directory (features stored in processed/ligand_embeddings/).
    scaler_dir : str
        Directory containing pre-fitted RobustScaler for descriptor normalization.
    ecfp_radius : int
        Radius used for computing ECFP fingerprints.
    fp_length : int
        Length of the fingerprint bit vector.
    calc_descriptors : bool
        Whether molecular descriptors should be computed.
    batch_size : int
        GPU batch size for MLP encoding.
    num_workers : int
        CPU workers per shard for RDKit feature computation.
    smiles_batch_size : int
        Number of SMILES per parallel batch within a shard.
    n_parallel : int
        Number of shards to process in parallel during feature extraction.
    shard_start : int
        First shard index to process.
    shard_end : int, optional
        Last shard index (exclusive). Defaults to all available shards.
    shard_indices : list of int, optional
        Explicit shard indices to process (overrides shard_start/shard_end).
    device : str
        Device used for inference.
    features_only : bool
        Only compute features, skip GPU encoding.
    encode_only : bool
        Only run GPU encoding (features must already exist).
    overwrite : bool
        Whether existing outputs should be recomputed.
    memory_limit : float, optional
        Minimum available RAM in GB before launching new shard jobs.
    input_file : str, optional
        Raw input file (.bz2/.gz/.tsv/.txt) to split into shards before processing.
    smiles_column : int
        Column index (0-based) of SMILES in the input file.
    shard_size : int
        Number of SMILES per shard when splitting.
    delimiter : str
        Field delimiter in input file.
    skip_header : bool
        Whether the input file has a header line to skip.
    save_f16 : bool
        Save embeddings as float16 (halves disk usage).
    vs_only : bool
        Only save the VS (protein-space) half of embeddings.
    """

    def __init__(
        self,
        checkpoint_path="./checkpoints/best_model",
        smiles_dir="./data/datasets/predict_datasets/enamine_large/info",
        output_dir="./data/datasets/predict_datasets/enamine_large",
        scaler_dir="data/common/scalers",
        ecfp_radius=2,
        fp_length=2048,
        calc_descriptors=True,
        batch_size=4096,
        num_workers=32,
        smiles_batch_size=1000,
        n_parallel=16,
        shard_start=0,
        shard_end=None,
        shard_indices=None,
        device="cuda:0",
        features_only=False,
        encode_only=False,
        overwrite=False,
        memory_limit=None,
        input_file=None,
        smiles_column=0,
        shard_size=1_000_000,
        delimiter="\t",
        skip_header=True,
        save_f16=False,
        vs_only=False,
    ):
        self.checkpoint_path = checkpoint_path
        self.smiles_dir = smiles_dir
        self.output_dir = output_dir
        self.scaler_dir = scaler_dir
        self.ecfp_radius = ecfp_radius
        self.fp_length = fp_length
        self.calc_descriptors = calc_descriptors
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.smiles_batch_size = smiles_batch_size
        self.n_parallel = n_parallel
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.shard_indices = shard_indices
        self.device = device
        self.features_only = features_only
        self.encode_only = encode_only
        self.overwrite = overwrite
        self.memory_limit = memory_limit
        self.input_file = input_file
        self.smiles_column = smiles_column
        self.shard_size = shard_size
        self.delimiter = delimiter
        self.skip_header = skip_header
        self.save_f16 = save_f16
        self.vs_only = vs_only

    def _get_shard_files(self):
        """
        Discover smiles_N.txt files and return sorted (index, path) pairs.

        Returns
        -------
        list of tuple
            Sorted list of (shard_index, file_path) pairs filtered by shard range.
        """
        pattern = re.compile(r"smiles_(\d+)\.txt$")
        files = []
        for fname in os.listdir(self.smiles_dir):
            match = pattern.match(fname)
            if match:
                idx = int(match.group(1))
                files.append((idx, os.path.join(self.smiles_dir, fname)))

        files.sort(key=lambda x: x[0])

        if self.shard_indices is not None:
            idx_set = set(self.shard_indices)
            files = [(idx, path) for idx, path in files if idx in idx_set]
        else:
            end = self.shard_end if self.shard_end is not None else len(files)
            files = [(idx, path) for idx, path in files if self.shard_start <= idx < end]

        return files

    def _setup_model(self):
        """
        Load the trained ligand encoder.

        Returns
        -------
        MLPEncoder
            Ligand encoder model with loaded weights in evaluation mode.
        """
        with open("configs/model/ligand_encoder/mlp.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg.pop("_target_", None)

        model = MLPEncoder(**cfg)
        state_dict = torch.load(
            f"{self.checkpoint_path}/ligand_encoder.pth", weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    def _is_shard_embedded(self, shard_idx):
        """
        Check if embeddings already exist for a shard.

        Parameters
        ----------
        shard_idx : int
            Index of the shard to check.

        Returns
        -------
        bool
            True if embedding file exists for this shard.
        """
        emb_dir = os.path.join(self.output_dir, "processed", "ligand_embeddings")
        for suffix in [f"embeddings_{shard_idx}.npz", f"embeddings_{shard_idx}_fp16.npz",
                       f"embeddings_{shard_idx}.npy", f"embeddings_{shard_idx}_fp16.npy"]:
            if os.path.exists(os.path.join(emb_dir, suffix)):
                return True
        return False

    @torch.no_grad()
    def _encode_shard(self, shard_idx, model):
        """
        Load features from .dat.gz and encode with the MLP.

        Parameters
        ----------
        shard_idx : int
            Index of the shard to encode.
        model : MLPEncoder
            Trained ligand encoder model.
        """
        emb_dir = os.path.join(self.output_dir, "processed", "ligand_embeddings")
        fp_name = f"ecfp{2 * self.ecfp_radius}_{self.fp_length}"
        metadata_path = os.path.join(emb_dir, f"metadata_{fp_name}_{shard_idx}.json")

        if not os.path.exists(metadata_path):
            print(f"[Shard {shard_idx}] No metadata found, cannot encode. Run features first.")
            return

        metadata = read_json(metadata_path)
        num_ligands = metadata["num_ligands"]
        fp_length = metadata["fingerprint_length"]

        fp_gz = os.path.join(emb_dir, f"{fp_name}_{shard_idx}.dat.gz")
        fp_dat = os.path.join(emb_dir, f"{fp_name}_{shard_idx}.dat")
        if os.path.exists(fp_gz):
            fingerprints = _load_features_from_gz(fp_gz, num_ligands, fp_length)
        elif os.path.exists(fp_dat):
            fingerprints = np.memmap(fp_dat, dtype="float32", mode="r", shape=(num_ligands, fp_length))
        else:
            print(f"[Shard {shard_idx}] No fingerprint file found.")
            return

        if self.calc_descriptors:
            desc_length = metadata["descriptor_length"]
            desc_gz = os.path.join(emb_dir, f"descriptors_{shard_idx}.dat.gz")
            desc_dat = os.path.join(emb_dir, f"descriptors_{shard_idx}.dat")
            if os.path.exists(desc_gz):
                descriptors = _load_features_from_gz(desc_gz, num_ligands, desc_length)
            elif os.path.exists(desc_dat):
                descriptors = np.memmap(desc_dat, dtype="float32", mode="r", shape=(num_ligands, desc_length))
            else:
                print(f"[Shard {shard_idx}] No descriptor file found.")
                return
            features = np.concatenate([fingerprints, descriptors], axis=1)
        else:
            features = fingerprints

        embeddings = []
        n = features.shape[0]
        n_batches = (n + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, n, self.batch_size), total=n_batches,
                      desc=f"[Shard {shard_idx}] GPU encoding"):
            batch = torch.from_numpy(features[i : i + self.batch_size]).to(self.device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        if self.vs_only:
            embeddings = embeddings[:, :embeddings.shape[1] // 2]

        precision = "float16" if self.save_f16 else "float32"
        if self.save_f16:
            embeddings = embeddings.astype(np.float16)
            emb_filename = f"embeddings_{shard_idx}_fp16.npz"
        else:
            emb_filename = f"embeddings_{shard_idx}.npz"

        np.savez_compressed(os.path.join(emb_dir, emb_filename), embeddings=embeddings)

        write_json(os.path.join(emb_dir, f"metadata_embeddings_{shard_idx}.json"), {
            "num_ligands": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1],
            "precision": precision,
            "vs_only": self.vs_only,
        })

        print(f"[Shard {shard_idx}] Encoded {embeddings.shape[0]} ligands -> {embeddings.shape} ({precision})")

        del features, fingerprints, embeddings
        if self.calc_descriptors:
            del descriptors
        gc.collect()

    def run(self):
        """
        Run the full embedding pipeline: split input, extract features, and encode.

        Executes up to three phases depending on configuration:
        splitting the input file into shards, parallel feature extraction,
        and sequential GPU encoding.
        """
        if self.input_file:
            print(f"\n{'='*60}")
            print(f"Phase 0: Splitting input file into shards")
            print(f"  Input: {self.input_file}")
            print(f"  Output: {self.smiles_dir}")
            print(f"  Shard size: {self.shard_size:,}")
            print(f"{'='*60}")
            split_input_file(
                self.input_file, self.smiles_dir, self.shard_size,
                self.smiles_column, self.delimiter, self.skip_header, self.overwrite,
            )

        shard_files = self._get_shard_files()
        if not shard_files:
            print("No shard files found.")
            return

        print(f"Found {len(shard_files)} shards to process (indices {shard_files[0][0]}–{shard_files[-1][0]})")

        if not self.encode_only:
            shards_needing_features = []
            emb_dir = os.path.join(self.output_dir, "processed", "ligand_embeddings")
            for idx, path in shard_files:
                fp_name = f"ecfp{2 * self.ecfp_radius}_{self.fp_length}"
                metadata_path = os.path.join(emb_dir, f"metadata_{fp_name}_{idx}.json")
                if self.overwrite or not os.path.exists(metadata_path):
                    shards_needing_features.append((idx, path))

            if shards_needing_features:
                print(f"\n{'='*60}")
                print(f"Phase 1: Feature extraction ({len(shards_needing_features)} shards, up to {self.n_parallel} parallel)")
                if self.memory_limit:
                    print(f"Memory limit: {self.memory_limit} GB (currently {_get_available_memory_gb():.0f} GB available)")
                print(f"{'='*60}")

                failed = []
                pending = list(shards_needing_features)
                pbar = tqdm(total=len(shards_needing_features), desc="Feature extraction (shards)")

                with ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
                    futures = {}

                    while pending or futures:
                        while pending and len(futures) < self.n_parallel:
                            if self.memory_limit:
                                available_gb = _get_available_memory_gb()
                                if available_gb < self.memory_limit:
                                    if futures:
                                        pbar.set_postfix_str(
                                            f"mem={available_gb:.0f}GB < {self.memory_limit}GB, waiting ({len(futures)} active)")
                                        break
                                    else:
                                        pbar.set_postfix_str(
                                            f"WARNING: mem={available_gb:.0f}GB low, no active jobs, proceeding")

                            idx, path = pending.pop(0)
                            future = executor.submit(
                                process_shard_features,
                                shard_idx=idx,
                                smiles_file=path,
                                output_dir=self.output_dir,
                                ecfp_radius=self.ecfp_radius,
                                fp_length=self.fp_length,
                                calc_descriptors=self.calc_descriptors,
                                num_workers=self.num_workers,
                                smiles_batch_size=self.smiles_batch_size,
                                scaler_dir=self.scaler_dir,
                                overwrite=self.overwrite,
                            )
                            futures[future] = idx

                        if not futures:
                            break

                        done = set()
                        for future in as_completed(futures):
                            done.add(future)
                            shard_idx = futures[future]
                            try:
                                future.result()
                                pbar.update(1)
                                avail = _get_available_memory_gb()
                                pbar.set_postfix_str(f"mem={avail:.0f}GB, {len(futures)-1} active")
                            except Exception as e:
                                failed.append(shard_idx)
                                pbar.update(1)
                                tqdm.write(f"  [Shard {shard_idx}] FAILED: {e}")
                            break

                        for f in done:
                            del futures[f]

                pbar.close()

                if failed:
                    print(f"\nWARNING: {len(failed)} shards failed: {sorted(failed)}")
            else:
                print("All shard features already computed.")

        if self.features_only:
            print("Features-only mode, skipping encoding.")
            return

        shards_needing_encoding = []
        for idx, _ in shard_files:
            if self.overwrite or not self._is_shard_embedded(idx):
                shards_needing_encoding.append(idx)

        if not shards_needing_encoding:
            print("All shard embeddings already computed.")
            return

        print(f"\n{'='*60}")
        print(f"Phase 2: GPU encoding ({len(shards_needing_encoding)} shards on {self.device})")
        print(f"{'='*60}")

        model = self._setup_model()

        for shard_idx in tqdm(shards_needing_encoding, desc="GPU encoding (shards)"):
            self._encode_shard(shard_idx, model)

        print(f"\nAll done. Encoded {len(shards_needing_encoding)} shards.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scalable ligand embedding pipeline for large SMILES datasets."
    )

    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model")
    parser.add_argument("--smiles_dir", type=str, default="./data/datasets/predict_datasets/enamine_large/info",
                        help="Directory containing smiles_N.txt files")
    parser.add_argument("--output_dir", type=str, default="./data/datasets/predict_datasets/enamine_large",
                        help="Base output directory (features stored in processed/ligand_embeddings/)")
    parser.add_argument("--scaler_dir", type=str, default="data/common/scalers")

    parser.add_argument("--ecfp_radius", type=int, default=2)
    parser.add_argument("--fp_length", type=int, default=2048)
    parser.add_argument("--no_descriptors", action="store_true", help="Disable descriptor computation")

    parser.add_argument("--batch_size", type=int, default=4096, help="GPU batch size for MLP encoding")
    parser.add_argument("--num_workers", type=int, default=32, help="CPU workers per shard for RDKit")
    parser.add_argument("--smiles_batch_size", type=int, default=1000, help="SMILES per joblib batch")
    parser.add_argument("--n_parallel", type=int, default=16, help="Number of shards to process in parallel")
    parser.add_argument("--memory_limit", type=float, default=200,
                        help="Minimum available RAM in GB. New shards won't launch if available memory drops below this.")

    parser.add_argument("--shard_start", type=int, default=0)
    parser.add_argument("--shard_end", type=int, default=None)
    parser.add_argument("--shard_indices", type=int, nargs="+", default=None,
                        help="Explicit shard indices (overrides start/end)")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Raw input file (.bz2/.gz/.tsv/.txt) to split into shards before processing")
    parser.add_argument("--smiles_column", type=int, default=0,
                        help="Column index (0-based) of SMILES in the input file")
    parser.add_argument("--shard_size", type=int, default=1_000_000,
                        help="Number of SMILES per shard when splitting")
    parser.add_argument("--delimiter", type=str, default="\t",
                        help="Field delimiter in input file")
    parser.add_argument("--no_header", action="store_true",
                        help="Input file has no header line to skip")
    parser.add_argument("--save_f16", action="store_true",
                        help="Save embeddings as float16 (halves disk usage)")
    parser.add_argument("--vs_only", action="store_true",
                        help="Only save the VS (protein-space) half of embeddings, skip the pocket-ranking half")

    parser.add_argument("--features_only", action="store_true", help="Only compute features, skip GPU encoding")
    parser.add_argument("--encode_only", action="store_true", help="Only run GPU encoding (features must exist)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if outputs exist")

    args = parser.parse_args()

    embedder = ShardedLigandEmbedder(
        checkpoint_path=args.checkpoint_path,
        smiles_dir=args.smiles_dir,
        output_dir=args.output_dir,
        scaler_dir=args.scaler_dir,
        ecfp_radius=args.ecfp_radius,
        fp_length=args.fp_length,
        calc_descriptors=not args.no_descriptors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        smiles_batch_size=args.smiles_batch_size,
        n_parallel=args.n_parallel,
        shard_start=args.shard_start,
        shard_end=args.shard_end,
        shard_indices=args.shard_indices,
        device=args.device,
        features_only=args.features_only,
        encode_only=args.encode_only,
        overwrite=args.overwrite,
        memory_limit=args.memory_limit,
        input_file=args.input_file,
        smiles_column=args.smiles_column,
        shard_size=args.shard_size,
        delimiter=args.delimiter,
        skip_header=not args.no_header,
        save_f16=args.save_f16,
        vs_only=args.vs_only,
    )

    embedder.run()
