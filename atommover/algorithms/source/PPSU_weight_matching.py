# source code in C (bttlThreshold): [PPSU2023](https://inria.hal.science/hal-04146298)

import ctypes
import io
import os
import platform
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
LIB_PATH = os.path.join(HERE, "..", "..", "extern", f"libmatching_for_PPSU{ext_suffix}")
lib = ctypes.CDLL(LIB_PATH)

# Load the shared library
try:
    lib = ctypes.CDLL(LIB_PATH)
except OSError as e:
    raise RuntimeError(f"Cannot load shared library at {LIB_PATH}: {e}")

# Platform-specific handling for accessing the standard C library
try:
    if platform.system() == "Windows":
        # On Windows, use msvcrt for C standard library functions
        libc = ctypes.CDLL("msvcrt")
    else:
        # On Unix-like systems (Linux, macOS), use None to get the default C library
        libc = ctypes.CDLL(None)
    
    try:
        stdout_fileno = sys.stdout.fileno()
        libc.fflush(None)  # Flush C stdio buffers
    except (AttributeError, OSError, io.UnsupportedOperation):
        # If fileno() is not available (e.g., in Jupyter), skip the flush
        sys.stdout.flush()  # Use Python's flush instead
        stdout_fileno = None
except (OSError, AttributeError) as e:
    # If we can't access the C library, define a no-op flush function
    print(f"[WARNING] Could not access C library for fflush: {e}")
    def noop_flush():
        sys.stdout.flush()  # Use Python's flush instead
    libc = type('MockLibc', (), {'fflush': lambda x: noop_flush()})()
    stdout_fileno = None

# Define the function signature for bttlThreshold
lib.bttlThreshold.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # col_ptrs
    ctypes.POINTER(ctypes.c_int),  # col_ids
    ctypes.POINTER(ctypes.c_double),  # col_vals
    ctypes.c_int,  # n
    ctypes.c_int,  # m
    ctypes.POINTER(ctypes.c_int),  # match
    ctypes.POINTER(ctypes.c_int),  # row_match
    ctypes.POINTER(ctypes.c_int),  # row_ptrs
    ctypes.POINTER(ctypes.c_int),  # row_ids
    ctypes.POINTER(ctypes.c_double),  # row_vals
    ctypes.POINTER(ctypes.c_int),  # fend_cols
    ctypes.POINTER(ctypes.c_int),  # fend_rows
    ctypes.c_int,  # lbapAlone
    ctypes.POINTER(ctypes.c_double),  # thrshld_g
    ctypes.c_int,  # sprankknown
]
lib.bttlThreshold.restype = ctypes.c_int  # Returns the number of iterations

def bttl_threshold(col_ptrs, col_ids, col_vals, n, m, sprankknown=0, lbapAlone=1):
    """
    Python wrapper for the bttlThreshold function in the shared library.

    Args:
        col_ptrs (list[int]): Column pointers (CSR format).
        col_ids (list[int]): Row indices (CSR format).
        col_vals (list[float]): Edge weights in CSR format.
        n (int): Number of columns.
        m (int): Number of rows.
        sprankknown (int): Structural rank of the matrix (default: 0).

    Returns:
        dict: Matching results, including column-to-row, row-to-column mappings, and threshold.
    """
    # Convert inputs to ctypes
    col_ptrs = (ctypes.c_int * len(col_ptrs))(*col_ptrs)
    col_ids = (ctypes.c_int * len(col_ids))(*col_ids)
    col_vals = (ctypes.c_double * len(col_vals))(*col_vals)
    match = (ctypes.c_int * n)(-1)  # Initialize match array with -1
    row_match = (ctypes.c_int * m)(-1)  # Initialize row_match array with -1
    row_ptrs = (ctypes.c_int * (m + 1))()  # Placeholder for row pointers
    row_ids = (ctypes.c_int * len(col_ids))()  # Placeholder for row indices
    row_vals = (ctypes.c_double * len(col_vals))()  # Placeholder for row values
    fend_cols = (ctypes.c_int * n)()  # Placeholder for fend_cols
    fend_rows = (ctypes.c_int * m)()  # Placeholder for fend_rows
    thrshld_g = ctypes.c_double()  # Threshold value

    # Call the C function
    iterations = lib.bttlThreshold(
        col_ptrs, col_ids, col_vals, ctypes.c_int(n), ctypes.c_int(m),
        match, row_match, row_ptrs, row_ids, row_vals,
        fend_cols, fend_rows, ctypes.c_int(lbapAlone), ctypes.byref(thrshld_g), sprankknown
    )

    # Return results
    return {
        "iterations": iterations,
        "match": list(match),
        "row_match": list(row_match),
        "threshold": thrshld_g.value,
    }
