"""
Tests for dequantization functions.
"""

import pytest
import numpy as np
from pyspz._dequantize import (
    decode_positions_24bit_signed,
    decode_alphas_u8,
    decode_colors_u8,
    decode_scales_u8exp,
    decode_rotations_v2,
    decode_rotations_v3,
    decode_sh_i8,
)


class TestDecodePositions:
    """Test cases for 24-bit signed position decoding."""
    
    def test_simple_positions(self):
        """Test decoding simple position data."""
        # Create test data: 2 points with known values
        # Point 1: (1.0, 2.0, 3.0) with fractional_bits=8
        # Point 2: (-1.0, -2.0, -3.0) with fractional_bits=8
        
        # Convert to 24-bit fixed point
        fractional_bits = 8
        p1_int = [int(1.0 * (2**8)), int(2.0 * (2**8)), int(3.0 * (2**8))]
        p2_int = [int(-1.0 * (2**8)), int(-2.0 * (2**8)), int(-3.0 * (2**8))]
        
        # Pack as little-endian 24-bit values
        buf = bytearray()
        for point in [p1_int, p2_int]:
            for val in point:
                # Convert to 24-bit signed and pack as 3 bytes
                val_24 = val & 0xFFFFFF
                buf.extend([val_24 & 0xFF, (val_24 >> 8) & 0xFF, (val_24 >> 16) & 0xFF])
        
        result = decode_positions_24bit_signed(bytes(buf), 2, fractional_bits)
        
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(result[1], [-1.0, -2.0, -3.0], rtol=1e-6)
    
    def test_wrong_buffer_size(self):
        """Test error handling for wrong buffer size."""
        buf = b"short"
        with pytest.raises(ValueError, match="Expected 18 bytes for positions"):
            decode_positions_24bit_signed(buf, 2, 8)


class TestDecodeAlphas:
    """Test cases for alpha decoding."""
    
    def test_alpha_logistic(self):
        """Test alpha decoding with logistic function."""
        # Test values: 0, 128, 255
        buf = bytes([0, 128, 255])
        result = decode_alphas_u8(buf, 3)
        
        assert result.shape == (3, 1)
        assert result.dtype == np.float32
        
        # Check logistic function: alpha = 1 / (1 + exp(-value))
        expected = np.array([[1.0 / (1.0 + np.exp(-0.0))],
                           [1.0 / (1.0 + np.exp(-128.0))],
                           [1.0 / (1.0 + np.exp(-255.0))]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_wrong_buffer_size(self):
        """Test error handling for wrong buffer size."""
        buf = b"short"
        with pytest.raises(ValueError, match="Expected 3 bytes for alphas"):
            decode_alphas_u8(buf, 3)


class TestDecodeColors:
    """Test cases for color decoding."""
    
    def test_color_normalization(self):
        """Test RGB color normalization."""
        # Test values: (0, 128, 255) for each point
        buf = bytes([0, 128, 255, 0, 128, 255])
        result = decode_colors_u8(buf, 2)
        
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        
        expected = np.array([[0.0, 128.0/255.0, 1.0],
                           [0.0, 128.0/255.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_wrong_buffer_size(self):
        """Test error handling for wrong buffer size."""
        buf = b"short"
        with pytest.raises(ValueError, match="Expected 6 bytes for colors"):
            decode_colors_u8(buf, 2)


class TestDecodeScales:
    """Test cases for scale decoding."""
    
    def test_scale_exponential(self):
        """Test scale decoding with exponential function."""
        # Test values: 0, 1, 2 (should become exp(0), exp(1), exp(2))
        buf = bytes([0, 1, 2, 0, 1, 2])
        result = decode_scales_u8exp(buf, 2)
        
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        
        expected = np.array([[np.exp(0), np.exp(1), np.exp(2)],
                           [np.exp(0), np.exp(1), np.exp(2)]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_wrong_buffer_size(self):
        """Test error handling for wrong buffer size."""
        buf = b"short"
        with pytest.raises(ValueError, match="Expected 6 bytes for scales"):
            decode_scales_u8exp(buf, 2)


class TestDecodeRotationsV2:
    """Test cases for rotation decoding (version 2)."""
    
    def test_rotation_v2_basic(self):
        """Test basic rotation decoding for v2."""
        # Test with simple values: (0, 0, 0) -> should give (1, 0, 0, 0)
        buf = bytes([0, 0, 0])
        result = decode_rotations_v2(buf, 1)
        
        assert result.shape == (1, 4)
        assert result.dtype == np.float32
        
        # Check that it's a valid quaternion
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_rotation_v2_normalization(self):
        """Test that rotations are properly normalized."""
        # Test with non-zero values
        buf = bytes([64, 64, 64])  # Should give a normalized quaternion
        result = decode_rotations_v2(buf, 1)
        
        assert result.shape == (1, 4)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_wrong_buffer_size(self):
        """Test error handling for wrong buffer size."""
        buf = b"short"
        with pytest.raises(ValueError, match="Expected 3 bytes for rotations v2"):
            decode_rotations_v2(buf, 1)


class TestDecodeRotationsV3:
    """Test cases for rotation decoding (version 3)."""
    
    def test_rotation_v3_basic(self):
        """Test basic rotation decoding for v3."""
        # Create a simple test case: imax=0, c1=0, c2=0, c3=0
        # This should result in quaternion (1, 0, 0, 0)
        value = 0  # imax=0, c1=0, c2=0, c3=0
        buf = value.to_bytes(4, 'little')
        result = decode_rotations_v3(buf, 1)
        
        assert result.shape == (1, 4)
        assert result.dtype == np.float32
        
        # Check that it's a valid quaternion
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_rotation_v3_normalization(self):
        """Test that v3 rotations are properly normalized."""
        # Test with non-zero values
        # imax=1, c1=100, c2=200, c3=300
        value = 1 | (100 << 2) | (200 << 12) | (300 << 22)
        buf = value.to_bytes(4, 'little')
        result = decode_rotations_v3(buf, 1)
        
        assert result.shape == (1, 4)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_wrong_buffer_size(self):
        """Test error handling for wrong buffer size."""
        buf = b"short"
        with pytest.raises(ValueError, match="Expected 4 bytes for rotations v3"):
            decode_rotations_v3(buf, 1)


class TestDecodeSH:
    """Test cases for spherical harmonics decoding."""
    
    def test_sh_degree_0(self):
        """Test SH decoding with degree 0 (no coefficients)."""
        buf = b""
        result = decode_sh_i8(buf, 2, 0)
        
        assert result.shape == (2, 0, 3)
        assert result.dtype == np.float32
    
    def test_sh_degree_1(self):
        """Test SH decoding with degree 1."""
        # 9 coefficients per point, 2 points = 18 bytes
        buf = bytes(range(18))
        result = decode_sh_i8(buf, 2, 1)
        
        assert result.shape == (2, 3, 3)  # 9 coefficients / 3 colors = 3 per color
        assert result.dtype == np.float32
        
        # Check normalization: values should be in [-1, 1]
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
    
    def test_sh_degree_2(self):
        """Test SH decoding with degree 2."""
        # 24 coefficients per point, 2 points = 48 bytes
        buf = bytes(range(48))
        result = decode_sh_i8(buf, 2, 2)
        
        assert result.shape == (2, 8, 3)  # 24 coefficients / 3 colors = 8 per color
        assert result.dtype == np.float32
    
    def test_sh_degree_3(self):
        """Test SH decoding with degree 3."""
        # 45 coefficients per point, 2 points = 90 bytes
        buf = bytes(range(90))
        result = decode_sh_i8(buf, 2, 3)
        
        assert result.shape == (2, 15, 3)  # 45 coefficients / 3 colors = 15 per color
        assert result.dtype == np.float32
    
    def test_wrong_buffer_size(self):
        """Test error handling for wrong buffer size."""
        buf = b"short"
        with pytest.raises(ValueError, match="Expected 18 bytes for SH coefficients"):
            decode_sh_i8(buf, 2, 1)
