"""
Core functionality for reading SPZ files.

This module handles the main file reading logic, including gzip decompression,
header parsing, and orchestration of the decoding process.
"""

import gzip
import io
from typing import Dict, Union, BinaryIO
import numpy as np

from ._header import SpzHeader
from ._dequantize import (
    decode_positions_24bit_signed,
    decode_alphas_u8,
    decode_colors_u8,
    decode_scales_u8exp,
    decode_rotations_v2,
    decode_rotations_v3,
    decode_sh_i8,
)


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
    """
    # Handle file path or file-like object
    if isinstance(source, str):
        with open(source, 'rb') as f:
            return _load_from_stream(f)
    else:
        return _load_from_stream(source)


def _load_from_stream(stream: BinaryIO) -> Dict[str, np.ndarray]:
    """Load SPZ data from a binary stream.
    
    Args:
        stream: Binary file-like object
        
    Returns:
        Dictionary of decoded Gaussian data
    """
    try:
        # Read and parse header
        header_data = stream.read(16)
        if len(header_data) != 16:
            raise ValueError(f"Could not read complete header: got {len(header_data)} bytes")
        
        header = SpzHeader.from_bytes(header_data)
        
        # Handle empty files
        if header.num_points == 0:
            return {
                'positions': np.zeros((0, 3), dtype=np.float32),
                'alphas': np.zeros((0, 1), dtype=np.float32),
                'colors': np.zeros((0, 3), dtype=np.float32),
                'scales': np.zeros((0, 3), dtype=np.float32),
                'rotations': np.zeros((0, 4), dtype=np.float32),
            }
        
        # Decompress the rest of the file
        compressed_data = stream.read()
        if not compressed_data:
            raise ValueError("No compressed data found after header")
        
        try:
            decompressed_data = gzip.decompress(compressed_data)
        except gzip.BadGzipFile as e:
            raise ValueError(f"Failed to decompress SPZ data: {e}")
        
        # Parse the decompressed data
        return _parse_body(decompressed_data, header)
        
    except Exception as e:
        raise RuntimeError(f"Error reading SPZ file: {e}") from e


def _parse_body(data: bytes, header: SpzHeader) -> Dict[str, np.ndarray]:
    """Parse the decompressed body of an SPZ file.
    
    Args:
        data: Decompressed binary data
        header: Parsed header information
        
    Returns:
        Dictionary of decoded Gaussian data
    """
    offset = 0
    num_points = header.num_points
    
    # Calculate expected sizes for each block
    positions_size = num_points * 3 * 3  # 3 components * 3 bytes each
    alphas_size = num_points * 1
    colors_size = num_points * 3
    scales_size = num_points * 3
    rotations_size = num_points * (3 if header.version == 2 else 4)
    sh_size = num_points * header.get_sh_coeff_count()
    
    # Read positions
    if offset + positions_size > len(data):
        raise ValueError(f"Insufficient data for positions: need {positions_size} bytes")
    positions_buf = data[offset:offset + positions_size]
    positions = decode_positions_24bit_signed(positions_buf, num_points, header.fractional_bits)
    offset += positions_size
    
    # Read alphas
    if offset + alphas_size > len(data):
        raise ValueError(f"Insufficient data for alphas: need {alphas_size} bytes")
    alphas_buf = data[offset:offset + alphas_size]
    alphas = decode_alphas_u8(alphas_buf, num_points)
    offset += alphas_size
    
    # Read colors
    if offset + colors_size > len(data):
        raise ValueError(f"Insufficient data for colors: need {colors_size} bytes")
    colors_buf = data[offset:offset + colors_size]
    colors = decode_colors_u8(colors_buf, num_points)
    offset += colors_size
    
    # Read scales
    if offset + scales_size > len(data):
        raise ValueError(f"Insufficient data for scales: need {scales_size} bytes")
    scales_buf = data[offset:offset + scales_size]
    scales = decode_scales_u8exp(scales_buf, num_points)
    offset += scales_size
    
    # Read rotations
    if offset + rotations_size > len(data):
        raise ValueError(f"Insufficient data for rotations: need {rotations_size} bytes")
    rotations_buf = data[offset:offset + rotations_size]
    if header.version == 2:
        rotations = decode_rotations_v2(rotations_buf, num_points)
    else:  # version == 3
        rotations = decode_rotations_v3(rotations_buf, num_points)
    offset += rotations_size
    
    # Read spherical harmonics (if present)
    result = {
        'positions': positions,
        'alphas': alphas,
        'colors': colors,
        'scales': scales,
        'rotations': rotations,
    }
    
    if header.sh_degree > 0:
        if offset + sh_size > len(data):
            raise ValueError(f"Insufficient data for spherical harmonics: need {sh_size} bytes")
        sh_buf = data[offset:offset + sh_size]
        sh_coeffs = decode_sh_i8(sh_buf, num_points, header.sh_degree)
        result['sh_coeffs'] = sh_coeffs
        offset += sh_size
    
    # Check for extra data
    if offset < len(data):
        extra_bytes = len(data) - offset
        # This might be acceptable in some cases, but log a warning
        print(f"Warning: {extra_bytes} extra bytes found after parsing complete")
    
    return result


def _validate_quaternions(quaternions: np.ndarray, tolerance: float = 1e-6) -> None:
    """Validate that quaternions are properly normalized.
    
    Args:
        quaternions: (N, 4) array of quaternions
        tolerance: Tolerance for normalization check
    """
    norms = np.linalg.norm(quaternions, axis=1)
    # Skip validation for zero quaternions (they will be normalized later)
    non_zero_mask = norms > tolerance
    if np.any(non_zero_mask):
        non_zero_norms = norms[non_zero_mask]
        if not np.allclose(non_zero_norms, 1.0, atol=tolerance):
            raise ValueError(f"Quaternions are not properly normalized: norms = {norms}")


def _validate_data_consistency(data: Dict[str, np.ndarray]) -> None:
    """Validate that all data arrays have consistent dimensions.
    
    Args:
        data: Dictionary of decoded data
    """
    if not data:
        return
    
    # Get the number of points from the first array
    num_points = len(data['positions'])
    
    # Check all arrays have the same number of points
    for key, array in data.items():
        if key == 'sh_coeffs':
            # SH coefficients can have different shape
            if len(array) != num_points:
                raise ValueError(f"Array '{key}' has {len(array)} points, expected {num_points}")
        else:
            if len(array) != num_points:
                raise ValueError(f"Array '{key}' has {len(array)} points, expected {num_points}")
    
    # Validate quaternion normalization
    if 'rotations' in data:
        _validate_quaternions(data['rotations'])
