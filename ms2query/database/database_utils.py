import io
import numpy as np


# ------------------------ helper functions ----------------------

def ndarray_to_blob(arr: np.ndarray) -> bytes:
    """Serialize a NumPy array (with dtype/shape) into bytes for SQLite BLOB."""
    with io.BytesIO() as f:
        np.save(f, arr, allow_pickle=False)
        return f.getvalue()


def blob_to_ndarray(blob: bytes) -> np.ndarray:
    """Deserialize a NumPy array (with dtype/shape) from SQLite BLOB."""
    # SQLite may return memoryview; ensure bytes
    b = bytes(blob)
    with io.BytesIO(b) as f:
        return np.load(f, allow_pickle=False)
