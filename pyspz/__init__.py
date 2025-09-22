"""
PySpz - Python library for reading SPZ (3D Gaussian Splatting) files.

This library provides functionality to read and decode SPZ files, which are
compressed 3D Gaussian Splatting data files. It is a Python port of the
JSpz Java library, maintaining binary compatibility with the original
C++ implementation from Niantic Labs.

The main entry point is the `load()` function, which reads SPZ files and
returns decoded data as NumPy arrays.
"""

from typing import Dict, Union, BinaryIO
import numpy as np

from ._core import load

__version__ = "0.1.0"
__all__ = ["load"]

# Type hints for the public API
def load(source: Union[str, BinaryIO]) -> Dict[str, np.ndarray]:
    """
    Load 3D Gaussian Splatting data from an SPZ file.
    
    Args:
        source: Path to .spz file or binary file-like object.
        
    Returns:
        Dictionary containing Gaussian attributes as NumPy arrays:
        - 'positions': (N, 3) float32, XYZ coordinates
        - 'alphas': (N, 1) float32, opacity values
        - 'colors': (N, 3) float32, RGB colors in [0,1] range
        - 'scales': (N, 3) float32, XYZ scales
        - 'rotations': (N, 4) float32, quaternions (w, x, y, z)
        - 'sh_coeffs': (N, K, 3) float32, spherical harmonics coefficients
          (only present if shDegree > 0)
          
    Raises:
        ValueError: If the file format is invalid or unsupported
        RuntimeError: If there are issues reading the file
        
    Example:
        >>> import pyspz
        >>> data = pyspz.load("model.spz")
        >>> print(f"Loaded {len(data['positions'])} Gaussians")
    """
    return _core.load(source)
