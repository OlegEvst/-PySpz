"""
Dequantization functions for SPZ file format.

This module contains vectorized functions for decoding various data types
from the SPZ binary format, including 24-bit signed integers, 8-bit values,
and quaternion rotations.
"""

import numpy as np
from typing import Tuple


def decode_positions_24bit_signed(buf: bytes, num_points: int, fractional_bits: int) -> np.ndarray:
    """Decode 24-bit signed fixed-point positions.
    
    Args:
        buf: Binary buffer containing 24-bit signed integers
        num_points: Number of points to decode
        fractional_bits: Number of fractional bits for fixed-point conversion
        
    Returns:
        (N, 3) float32 array of XYZ positions
    """
    expected_size = num_points * 3 * 3  # 3 components * 3 bytes each
    if len(buf) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes for positions, got {len(buf)}")
    
    # Reshape to (N, 3, 3) for easier processing
    data = np.frombuffer(buf, dtype=np.uint8).reshape(num_points, 3, 3)
    
    # Convert 3-byte little-endian to 32-bit signed integers
    # Each 24-bit value is stored as 3 bytes in little-endian order
    positions = np.zeros((num_points, 3), dtype=np.float32)
    
    for i in range(num_points):
        for j in range(3):
            # Read 3 bytes as little-endian
            byte0, byte1, byte2 = data[i, j, :]
            
            # Combine into 24-bit value
            raw_value = int(byte0) | (int(byte1) << 8) | (int(byte2) << 16)
            
            # Sign extend from 24-bit to 32-bit
            if raw_value & 0x800000:  # Check sign bit
                raw_value -= 1 << 24  # Sign extend
            
            # Convert to float and apply fractional scaling
            positions[i, j] = raw_value / (2 ** fractional_bits)
    
    return positions


def decode_alphas_u8(buf: bytes, num_points: int) -> np.ndarray:
    """Decode 8-bit alpha values using logistic function.
    
    Args:
        buf: Binary buffer containing 8-bit unsigned integers
        num_points: Number of points to decode
        
    Returns:
        (N, 1) float32 array of alpha values
    """
    expected_size = num_points
    if len(buf) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes for alphas, got {len(buf)}")
    
    # Read as uint8 and convert to float
    alphas_int = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    
    # Apply logistic function: alpha = 1 / (1 + exp(-value))
    alphas = 1.0 / (1.0 + np.exp(-alphas_int))
    
    return alphas.reshape(-1, 1)


def decode_colors_u8(buf: bytes, num_points: int) -> np.ndarray:
    """Decode 8-bit RGB colors.
    
    Args:
        buf: Binary buffer containing 8-bit unsigned integers
        num_points: Number of points to decode
        
    Returns:
        (N, 3) float32 array of RGB colors in [0,1] range
    """
    expected_size = num_points * 3
    if len(buf) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes for colors, got {len(buf)}")
    
    # Read as uint8 and normalize to [0,1]
    colors_int = np.frombuffer(buf, dtype=np.uint8).reshape(num_points, 3)
    colors = colors_int.astype(np.float32) / 255.0
    
    return colors


def decode_scales_u8exp(buf: bytes, num_points: int) -> np.ndarray:
    """Decode 8-bit scales using exponential function.
    
    Args:
        buf: Binary buffer containing 8-bit unsigned integers
        num_points: Number of points to decode
        
    Returns:
        (N, 3) float32 array of scale values
    """
    expected_size = num_points * 3
    if len(buf) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes for scales, got {len(buf)}")
    
    # Read as uint8 and apply exponential function
    scales_int = np.frombuffer(buf, dtype=np.uint8).reshape(num_points, 3)
    scales = np.exp(scales_int.astype(np.float32))
    
    return scales


def decode_rotations_v2(buf: bytes, num_points: int) -> np.ndarray:
    """Decode rotations for SPZ version 2 (3x8-bit signed).
    
    Args:
        buf: Binary buffer containing 8-bit signed integers
        num_points: Number of points to decode
        
    Returns:
        (N, 4) float32 array of quaternions (w, x, y, z)
    """
    expected_size = num_points * 3
    if len(buf) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes for rotations v2, got {len(buf)}")
    
    # Read as int8 and normalize to [-1, 1]
    rotations_int = np.frombuffer(buf, dtype=np.int8).reshape(num_points, 3)
    q_xyz = rotations_int.astype(np.float32) / 127.0
    
    # Compute w component: w = sqrt(1 - (x^2 + y^2 + z^2))
    q_w_squared = 1.0 - np.sum(q_xyz**2, axis=1)
    q_w = np.sqrt(np.maximum(0.0, q_w_squared))  # Protect against negative values
    
    # Combine into quaternions (w, x, y, z)
    quaternions = np.column_stack([q_w, q_xyz])
    
    # Normalize to unit length
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / norms
    
    return quaternions


def decode_rotations_v3(buf: bytes, num_points: int) -> np.ndarray:
    """Decode rotations for SPZ version 3 (compressed quaternion).
    
    Args:
        buf: Binary buffer containing 4-byte compressed quaternions
        num_points: Number of points to decode
        
    Returns:
        (N, 4) float32 array of quaternions (w, x, y, z)
    """
    expected_size = num_points * 4
    if len(buf) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes for rotations v3, got {len(buf)}")
    
    # Read as uint32 for bit manipulation
    data = np.frombuffer(buf, dtype=np.uint32)
    
    quaternions = np.zeros((num_points, 4), dtype=np.float32)
    
    for i in range(num_points):
        value = data[i]
        
        # Extract 2-bit index of largest component
        imax = value & 0x3
        value >>= 2
        
        # Extract three 10-bit signed values
        c1 = value & 0x3FF
        c2 = (value >> 10) & 0x3FF
        c3 = (value >> 20) & 0x3FF
        
        # Sign extend 10-bit values
        if c1 & 0x200:
            c1 -= 1 << 10
        if c2 & 0x200:
            c2 -= 1 << 10
        if c3 & 0x200:
            c3 -= 1 << 10
        
        # Normalize to [-1, 1]
        q_components = np.array([c1, c2, c3], dtype=np.float32) / 511.0
        
        # Insert components into quaternion, skipping the largest
        quat = np.zeros(4)
        comp_idx = 0
        for j in range(4):
            if j != imax:
                quat[j] = q_components[comp_idx]
                comp_idx += 1
        
        # Compute the missing component
        sum_squares = np.sum(quat**2)
        quat[imax] = np.sqrt(max(0.0, 1.0 - sum_squares))
        
        # Normalize
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm
        
        quaternions[i] = quat
    
    return quaternions


def decode_sh_i8(buf: bytes, num_points: int, sh_degree: int) -> np.ndarray:
    """Decode spherical harmonics coefficients.
    
    Args:
        buf: Binary buffer containing 8-bit signed integers
        num_points: Number of points to decode
        sh_degree: Degree of spherical harmonics (0-3)
        
    Returns:
        (N, K, 3) float32 array of SH coefficients, where K depends on sh_degree
    """
    if sh_degree == 0:
        return np.zeros((num_points, 0, 3), dtype=np.float32)
    
    # Calculate number of coefficients per point
    sh_counts = {1: 9, 2: 24, 3: 45}
    coeffs_per_point = sh_counts[sh_degree]
    expected_size = num_points * coeffs_per_point
    
    if len(buf) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes for SH coefficients, got {len(buf)}")
    
    # Read as int8 and normalize to [-1, 1]
    sh_int = np.frombuffer(buf, dtype=np.int8).reshape(num_points, coeffs_per_point)
    sh_float = sh_int.astype(np.float32) / 128.0
    
    # Reshape to (N, K, 3) where K = coeffs_per_point / 3
    # SH coefficients are organized as "color as inner axis"
    coeffs_per_color = coeffs_per_point // 3
    sh_coeffs = sh_float.reshape(num_points, coeffs_per_color, 3)
    
    return sh_coeffs


def _read_24bit_signed_vectorized(data: np.ndarray) -> np.ndarray:
    """Vectorized reading of 24-bit signed integers.
    
    Args:
        data: (N, 3) array of uint8 values representing 24-bit integers
        
    Returns:
        (N,) array of float32 values
    """
    # Combine 3 bytes into 24-bit values
    raw_values = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
    
    # Sign extend from 24-bit to 32-bit
    mask = raw_values & 0x800000
    raw_values = np.where(mask, raw_values - (1 << 24), raw_values)
    
    return raw_values.astype(np.float32)


def _read_10bit_signed_vectorized(data: np.ndarray) -> np.ndarray:
    """Vectorized reading of 10-bit signed integers from packed data.
    
    Args:
        data: Array of packed 10-bit values
        
    Returns:
        Array of float32 values
    """
    # Sign extend 10-bit values
    mask = data & 0x200
    data = np.where(mask, data - (1 << 10), data)
    
    return data.astype(np.float32)
