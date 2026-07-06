import os
import re
import bz2
import gzip
import time
import heapq
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from conglude.utils.common import read_list_from_txt, read_json
from embed_proteins import ProteinEmbedder


class LargeLibraryScreener:
    """
    Screen a pre-encoded ligand library against one or more proteins.

    Computes protein embeddings via ProteinEmbedder (or loads existing ones),
    then scores all ligands shard-by-shard using cosine similarity in
    the protein embedding space.

    Parameters
    ----------
    protein_dataset_dir : str
        Dataset directory containing info/protein_ids.txt and raw PDB files.
    ligand_emb_dir : str
        Directory containing pre-computed shard embeddings
        (processed/ligand_embeddings/embeddings_N.npz).
    smiles_dir : str
        Directory containing smiles_N.txt shard files.
    output_dir : str
        Directory where score files will be written.
    checkpoint_path : str
        Directory containing trained model weights.
    protein_results_dir : str, optional
        If provided, load existing protein embeddings from this directory
        instead of re-computing them.
    pdb_dir : str, optional
        Directory containing raw PDB files for protein embedding.
    input_file : str, optional
        Original input file used to create shards. Required if id_column is set.
        Used to extract compound IDs and verify SMILES alignment with shard files.
    id_column : int, optional
        Column index (0-based) of compound IDs in the original input file.
    smiles_column : int
        Column index (0-based) of SMILES in the original input file.
    input_delimiter : str
        Field delimiter in the original input file.
    skip_header : bool
        Whether the original input file has a header line to skip.
    num_shards : int, optional
        Number of shards to process. If None, auto-detected from smiles_dir.
    shard_start : int
        First shard index to process.
    shard_end : int, optional
        Last shard index (exclusive). Defaults to num_shards.
    prefetch_workers : int
        Number of threads for prefetching shard embeddings.
    top_k : int
        Number of top-scoring ligands to save per protein (0 to disable).
    compress : bool
        Whether to bz2-compress the output CSV.
    compresslevel : int
        Compression level for bz2 output (1-9).
    batch_size : int
        Batch size for protein embedding computation.
    device : str
        Device used for protein embedding inference.
    """

    DELIMITER = ";"

    def __init__(
        self,
        protein_dataset_dir,
        ligand_emb_dir,
        smiles_dir,
        output_dir,
        checkpoint_path="./checkpoints/best_model",
        protein_results_dir=None,
        pdb_dir=None,
        input_file=None,
        id_column=None,
        smiles_column=0,
        input_delimiter="\t",
        skip_header=True,
        num_shards=None,
        shard_start=0,
        shard_end=None,
        top_k=10_000,
        prefetch_workers=4,
        compress=True,
        compresslevel=9,
        batch_size=64,
        device="cuda:0",
    ):
        self.protein_dataset_dir = protein_dataset_dir
        self.ligand_emb_dir = ligand_emb_dir
        self.smiles_dir = smiles_dir
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.protein_results_dir = protein_results_dir
        self.pdb_dir = pdb_dir
        self.input_file = input_file
        self.id_column = id_column
        self.smiles_column = smiles_column
        self.input_delimiter = input_delimiter
        self.skip_header = skip_header
        self.top_k = top_k
        self.prefetch_workers = prefetch_workers
        self.compress = compress
        self.compresslevel = compresslevel
        self.batch_size = batch_size
        self.device = device

        self.shard_start = shard_start
        self.shard_end = shard_end
        self.num_shards = num_shards

        if self.id_column is not None and self.input_file is None:
            raise ValueError("--input_file is required when --id_column is set")

    def _detect_shards(self):
        """
        Discover available shard files and determine the processing range.

        Returns
        -------
        list of int
            Sorted shard indices to process.
        """
        pattern = re.compile(r"smiles_(\d+)\.txt$")
        all_indices = sorted(
            int(m.group(1))
            for f in os.listdir(self.smiles_dir)
            if (m := pattern.match(f))
        )

        if not all_indices:
            raise FileNotFoundError(f"No smiles_N.txt files found in {self.smiles_dir}")

        if self.num_shards is not None:
            all_indices = [i for i in all_indices if i < self.num_shards]

        end = self.shard_end if self.shard_end is not None else max(all_indices) + 1
        return [i for i in all_indices if self.shard_start <= i < end]

    def _get_protein_embeddings(self):
        """
        Load or compute protein embeddings.

        Returns
        -------
        protein_names : list of str
            Protein identifiers.
        protein_embeddings : np.ndarray
            Normalized protein embeddings of shape (num_proteins, embedding_dim).
        """
        if self.protein_results_dir is not None:
            emb_path = os.path.join(self.protein_results_dir, "embeddings", "protein_embeddings.npy")
            names_path = os.path.join(self.protein_results_dir, "embeddings", "protein_names.txt")
            embeddings = np.load(emb_path).astype(np.float32)
            names = read_list_from_txt(names_path)
        else:
            embedder = ProteinEmbedder(
                checkpoint_path=self.checkpoint_path,
                dataset_dir=self.protein_dataset_dir,
                pdb_dir=self.pdb_dir,
                results_dir=os.path.join(self.output_dir, "protein_embeddings"),
                batch_size=self.batch_size,
                save_embeddings=True,
                device=self.device,
            )
            names, embeddings, _, _ = embedder.embed()
            embeddings = embeddings.cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms

        return names, embeddings

    def _load_shard_embeddings(self, shard_idx):
        """
        Load pre-computed ligand embeddings for a single shard.

        Reads the embedding metadata to determine precision and whether
        only the VS half was saved. If full embeddings (VS + pocket-ranking)
        are stored, only the first half (protein-space) is used.

        Parameters
        ----------
        shard_idx : int
            Index of the shard to load.

        Returns
        -------
        np.ndarray
            Normalized VS ligand embeddings of shape (num_ligands, vs_dim).
        """
        emb_dir = os.path.join(self.ligand_emb_dir, "processed", "ligand_embeddings")

        metadata_path = os.path.join(emb_dir, f"metadata_embeddings_{shard_idx}.json")
        if os.path.exists(metadata_path):
            metadata = read_json(metadata_path)
            vs_only = metadata.get("vs_only", False)
            is_fp16 = metadata.get("precision", "float32") == "float16"
        else:
            vs_only = False
            is_fp16 = False

        if is_fp16:
            filename = f"embeddings_{shard_idx}_fp16.npz"
        else:
            filename = f"embeddings_{shard_idx}.npz"

        path = os.path.join(emb_dir, filename)
        if not os.path.exists(path):
            alt = f"embeddings_{shard_idx}_fp16.npz" if not is_fp16 else f"embeddings_{shard_idx}.npz"
            path = os.path.join(emb_dir, alt)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"No embedding file found for shard {shard_idx} in {emb_dir}"
                )

        with np.load(path) as data:
            emb = data["embeddings"].astype(np.float32)

        if not vs_only:
            emb = emb[:, :emb.shape[1] // 2]

        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb /= norms

        return emb

    def _load_shard_smiles(self, shard_idx):
        """
        Load SMILES strings for a single shard.

        Parameters
        ----------
        shard_idx : int
            Index of the shard to load.

        Returns
        -------
        list of str
            SMILES strings.
        """
        path = os.path.join(self.smiles_dir, f"smiles_{shard_idx}.txt")
        return read_list_from_txt(path)

    def _open_input_file(self):
        """
        Open the original input file for reading IDs.

        Returns
        -------
        file object
            Readable text file handle (supports .bz2 and .gz).
        """
        if self.input_file.endswith(".bz2"):
            fh = bz2.open(self.input_file, "rt")
        elif self.input_file.endswith(".gz"):
            fh = gzip.open(self.input_file, "rt")
        else:
            fh = open(self.input_file, "r")

        if self.skip_header:
            fh.readline()

        return fh

    def _read_shard_from_input(self, fh, n_lines, shard_smiles):
        """
        Read n_lines from the original input file, extract IDs and verify SMILES.

        Parameters
        ----------
        fh : file object
            Open file handle positioned at the start of this shard's rows.
        n_lines : int
            Number of lines to read (must match shard size).
        shard_smiles : list of str
            SMILES from the shard file, used for alignment verification.

        Returns
        -------
        list of str
            Compound IDs extracted from the input file.
        """
        ids = []
        max_col = max(self.id_column, self.smiles_column)

        for i in range(n_lines):
            line = fh.readline()
            if not line:
                raise RuntimeError(
                    f"Input file ended prematurely (expected {n_lines} lines, got {i})"
                )
            fields = line.rstrip("\n").split(self.input_delimiter, max_col + 1)
            ids.append(fields[self.id_column].strip())

            input_smiles = fields[self.smiles_column].strip().strip('"').split(" |")[0].strip()
            if input_smiles != shard_smiles[i]:
                raise RuntimeError(
                    f"SMILES mismatch at row {i}: "
                    f"input='{input_smiles[:80]}' vs shard='{shard_smiles[i][:80]}'"
                )

        return ids

    def _open_output(self, path):
        """
        Open an output file, optionally with bz2 compression.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        file object
            Writable text file handle.
        """
        if self.compress:
            if not path.endswith(".bz2"):
                path += ".bz2"
            return bz2.open(path, "wt", compresslevel=self.compresslevel)
        return open(path, "w")

    def _save_top_k(self, heaps, protein_names, has_ids):
        """
        Write per-protein top-k files from accumulated heaps.

        Parameters
        ----------
        heaps : dict
            Mapping of protein name to list of (score, row_string) tuples.
        protein_names : list of str
            Protein identifiers.
        has_ids : bool
            Whether the output includes an ID column.
        """
        sep = self.DELIMITER

        for j, name in enumerate(protein_names):
            heap = heaps[name]
            heap.sort(reverse=True)

            top_k_path = os.path.join(self.output_dir, f"vs_top{self.top_k}_{name}.csv")

            header_parts = []
            if has_ids:
                header_parts.append("id")
            header_parts.append("smiles")
            header_parts.append(f"score_{name}")

            with open(top_k_path, "w") as f:
                f.write(sep.join(header_parts) + "\n")
                for score, row_data in heap:
                    f.write(f"{row_data}{sep}{score:.6f}\n")

            print(f"  Top-{len(heap)} for {name}: {top_k_path}")
            if heap:
                print(f"    Score range: {heap[-1][0]:.6f} to {heap[0][0]:.6f}")

    def screen(self):
        """
        Score all ligands in the library against each protein.

        Writes a semicolon-delimited file with columns:
        [id;] smiles; score_<protein1>; score_<protein2>; ...
        The id column is included only when --input_file and --id_column are set.
        Additionally writes per-protein top-k files if top_k > 0.
        """
        sep = self.DELIMITER

        print("Computing protein embeddings...")
        protein_names, protein_embeddings = self._get_protein_embeddings()
        print(f"Proteins: {len(protein_names)} ({', '.join(protein_names)})")
        print(f"Protein embedding shape: {protein_embeddings.shape}")

        shard_indices = self._detect_shards()
        print(f"Shards to process: {len(shard_indices)} "
              f"(indices {shard_indices[0]}–{shard_indices[-1]})")

        has_ids = self.input_file is not None and self.id_column is not None

        os.makedirs(self.output_dir, exist_ok=True)
        output_name = "vs_scores.csv" + (".bz2" if self.compress else "")
        output_path = os.path.join(self.output_dir, output_name)

        score_columns = [f"score_{name}" for name in protein_names]
        header_parts = []
        if has_ids:
            header_parts.append("id")
        header_parts.append("smiles")
        header_parts.extend(score_columns)
        header = sep.join(header_parts) + "\n"

        total_ligands = 0
        score_stats = {name: {"min": np.inf, "max": -np.inf, "sum": 0.0}
                       for name in protein_names}
        heaps = {name: [] for name in protein_names}
        start_time = time.time()

        input_fh = self._open_input_file() if has_ids else None

        try:
            with self._open_output(output_path) as out_f:
                out_f.write(header)

                with ThreadPoolExecutor(max_workers=self.prefetch_workers) as executor:
                    futures = {}
                    prefetch_limit = min(self.prefetch_workers, len(shard_indices))

                    for i in range(prefetch_limit):
                        idx = shard_indices[i]
                        futures[idx] = executor.submit(self._load_shard_embeddings, idx)
                    next_submit = prefetch_limit

                    for progress, shard_idx in enumerate(shard_indices):
                        smiles = self._load_shard_smiles(shard_idx)
                        ligand_emb = futures[shard_idx].result()
                        del futures[shard_idx]

                        if next_submit < len(shard_indices):
                            next_idx = shard_indices[next_submit]
                            futures[next_idx] = executor.submit(
                                self._load_shard_embeddings, next_idx
                            )
                            next_submit += 1

                        assert len(smiles) == ligand_emb.shape[0], (
                            f"Shard {shard_idx}: SMILES count ({len(smiles)}) "
                            f"!= embedding count ({ligand_emb.shape[0]})"
                        )

                        ids = None
                        if has_ids:
                            ids = self._read_shard_from_input(
                                input_fh, len(smiles), smiles
                            )

                        scores = ligand_emb @ protein_embeddings.T

                        lines = []
                        for i in range(len(smiles)):
                            score_str = sep.join(f"{scores[i, j]:.6f}"
                                                 for j in range(scores.shape[1]))
                            if ids is not None:
                                lines.append(f"{ids[i]}{sep}{smiles[i]}{sep}{score_str}")
                            else:
                                lines.append(f"{smiles[i]}{sep}{score_str}")
                        out_f.write("\n".join(lines) + "\n")

                        if self.top_k > 0:
                            for j, name in enumerate(protein_names):
                                heap = heaps[name]
                                for i in range(len(smiles)):
                                    score = float(scores[i, j])
                                    row_data = f"{ids[i]}{sep}{smiles[i]}" if ids else smiles[i]
                                    if len(heap) < self.top_k:
                                        heapq.heappush(heap, (score, row_data))
                                    elif score > heap[0][0]:
                                        heapq.heapreplace(heap, (score, row_data))

                        for j, name in enumerate(protein_names):
                            col = scores[:, j]
                            score_stats[name]["min"] = min(score_stats[name]["min"], col.min())
                            score_stats[name]["max"] = max(score_stats[name]["max"], col.max())
                            score_stats[name]["sum"] += col.sum()

                        total_ligands += len(smiles)

                        elapsed = time.time() - start_time
                        rate = (progress + 1) / elapsed
                        eta = (len(shard_indices) - progress - 1) / rate
                        print(
                            f"  [{progress + 1}/{len(shard_indices)}] "
                            f"shard {shard_idx} ({len(smiles)} ligands) | "
                            f"elapsed: {elapsed / 3600:.2f}h | "
                            f"ETA: {eta / 3600:.2f}h"
                        )
        finally:
            if input_fh is not None:
                input_fh.close()

        total_time = time.time() - start_time
        print(f"\nDone in {total_time / 3600:.2f}h")
        print(f"Total ligands scored: {total_ligands:,}")
        for name in protein_names:
            stats = score_stats[name]
            mean = stats["sum"] / total_ligands if total_ligands > 0 else 0
            print(f"  {name}: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={mean:.6f}")
        print(f"Output: {output_path}")

        if self.top_k > 0:
            print(f"\nSaving top-{self.top_k} predictions per protein...")
            self._save_top_k(heaps, protein_names, has_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Screen a pre-encoded ligand library against proteins using ConGLUDe."
    )

    parser.add_argument("--protein_dataset_dir", type=str, required=True,
                        help="Dataset directory with info/protein_ids.txt and PDB files.")
    parser.add_argument("--ligand_emb_dir", type=str, required=True,
                        help="Directory containing pre-computed shard embeddings.")
    parser.add_argument("--smiles_dir", type=str, required=True,
                        help="Directory containing smiles_N.txt shard files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output score files.")

    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model",
                        help="Directory containing trained model weights.")
    parser.add_argument("--protein_results_dir", type=str, default=None,
                        help="Load existing protein embeddings from this directory instead of computing them.")
    parser.add_argument("--pdb_dir", type=str, default=None,
                        help="Directory containing raw PDB files for protein embedding.")

    parser.add_argument("--input_file", type=str, default=None,
                        help="Original input file for extracting compound IDs. Required if --id_column is set.")
    parser.add_argument("--id_column", type=int, default=None,
                        help="Column index (0-based) of compound IDs in the original input file.")
    parser.add_argument("--smiles_column", type=int, default=0,
                        help="Column index (0-based) of SMILES in the original input file (for verification).")
    parser.add_argument("--input_delimiter", type=str, default="\t",
                        help="Field delimiter in the original input file.")
    parser.add_argument("--no_header", action="store_true",
                        help="Original input file has no header line to skip.")

    parser.add_argument("--top_k", type=int, default=10_000,
                        help="Save the top-k scoring ligands per protein (0 to disable).")

    parser.add_argument("--num_shards", type=int, default=None,
                        help="Number of shards to process (auto-detected if None).")
    parser.add_argument("--shard_start", type=int, default=0,
                        help="First shard index to process.")
    parser.add_argument("--shard_end", type=int, default=None,
                        help="Last shard index (exclusive).")
    parser.add_argument("--prefetch_workers", type=int, default=4,
                        help="Threads for prefetching shard embeddings.")

    parser.add_argument("--no_compress", action="store_true",
                        help="Write output as plain CSV instead of bz2-compressed.")
    parser.add_argument("--compresslevel", type=int, default=9,
                        help="bz2 compression level (1-9).")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for protein embedding computation.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for protein embedding inference.")

    args = parser.parse_args()

    screener = LargeLibraryScreener(
        protein_dataset_dir=args.protein_dataset_dir,
        ligand_emb_dir=args.ligand_emb_dir,
        smiles_dir=args.smiles_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        protein_results_dir=args.protein_results_dir,
        pdb_dir=args.pdb_dir,
        input_file=args.input_file,
        id_column=args.id_column,
        smiles_column=args.smiles_column,
        input_delimiter=args.input_delimiter,
        skip_header=not args.no_header,
        num_shards=args.num_shards,
        shard_start=args.shard_start,
        shard_end=args.shard_end,
        top_k=args.top_k,
        prefetch_workers=args.prefetch_workers,
        compress=not args.no_compress,
        compresslevel=args.compresslevel,
        batch_size=args.batch_size,
        device=args.device,
    )

    screener.screen()
