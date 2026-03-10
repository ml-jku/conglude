import json
from typing import Any, Callable, List, Optional
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing



def read_list_from_txt(
    path: str
) -> List[str]:
    """
    Read a text file and return its contents as a list of strings.

    Parameters
    ----------
    path: str
        Path to the input text file.

    Returns
    -------
    List[str]
        A list containing one string per line in the file,
        without trailing newline characters.
    """

    with open(path, 'r') as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]



def write_list_to_txt(
    path: str, 
    str_list: List[str]
) -> None:
    """
    Write a list of strings to a text file.

    Parameters
    ----------
    path: str
        Path to the output text file.
    str_list: List[str]
        List of strings to write to the file.
    """

    with open(path, 'w') as file:
        new_lines = [f'{item}\n' for item in str_list]
        file.writelines(new_lines)



def read_json(
    path: str
) -> Any:
    """
    Read a JSON file and return its contents as a Python object.

    Parameters
    ----------
    path: str
        Path to the JSON file.

    Returns
    -------
    Any
        The Python object resulting from parsing the JSON file (typically a dictionary or list).
    """

    with open(path, 'r') as file:
        json_object = json.load(file)
    return json_object



def write_json(
    path: str, 
    json_object: Any
) -> None:
    """
    Write a Python object to a JSON file.

    Parameters
    ----------
    path: str
        Path to the output JSON file.
    json_object: Any
        Object to serialize and write.
    """

    with open(path, 'w') as file:
        json.dump(json_object, file)



def execute_in_parallel(
    func: Callable[..., Any],
    variable_args: List[Any],
    constant_args: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    max_gpu_jobs: int = 0,
    verbose: int = 1,
    desc: Optional[str] = None,
) -> List[Any]:
    """
    Parallel execution of a function over a list of inputs, with optional GPU concurrency control.

    Parameters
    ----------
    func: Callable[..., Any]
        The function to execute. It should accept at least one argument (from variable_args). If GPU locking is needed, it must accept a '_gpu_semaphore' keyword argument.
    variable_args: List[Any]
        List of variable arguments (one per call).
    constant_args: Optional[dict]
        Additional constant keyword arguments passed to func. Defaults to None.
    n_jobs: Optional[int]
        Number of parallel jobs. Defaults to None (all CPUs).
    max_gpu_jobs: int
        Max concurrent GPU users. 0 disables GPU locking. Defaults to 0.
    verbose: Optional[int]
        Verbosity level for joblib. Defaults to 1.
    desc: Optional[str]
        Description for tqdm progress bar. Defaults to None.

    Returns
    -------
    results: List[Any] 
        List of results from each function call.
    """

    # Ensure constant_args is always a dictionary
    constant_args = constant_args or {}

    # If max_gpu_jobs > 0, a shared semaphore is created that limits how many parallel processes can access the GPU simultaneously.
    gpu_semaphore = None
    if max_gpu_jobs > 0:
        manager = multiprocessing.Manager()
        gpu_semaphore = manager.BoundedSemaphore(max_gpu_jobs)

    def wrapper(var_arg):
        """
        Small wrapper function executed in parallel.
        It injects the GPU semaphore into the function call only if GPU concurrency control is enabled.
        """
        if gpu_semaphore is not None:
            return func(var_arg, _gpu_semaphore=gpu_semaphore, **constant_args)
        else:
            return func(var_arg, **constant_args)

    # Execute the wrapper function in parallel using joblib
    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
        delayed(wrapper)(arg) for arg in tqdm(variable_args, desc=desc, total=len(variable_args))
    )

    return results