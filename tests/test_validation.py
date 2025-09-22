"""
End-to-end validation tests for SPZ file loading.

These tests validate the complete pipeline from SPZ file reading to
decoded data, ensuring compatibility with the original JSpz library.
"""

import pytest
import numpy as np
import gzip
import io
from pyspz import load
from pyspz._header import SpzHeader


class TestValidation:
    """End-to-end validation tests."""
    
    def test_minimal_spz_file(self):
        """Test loading a minimal SPZ file with known data."""
        # Create a minimal SPZ file with 2 points
        header = SpzHeader(
            magic=0x5053474E,
            version=2,
            num_points=2,
            sh_degree=0,
            fractional_bits=8,
            flags=0,
            reserved=0
        )
        
        # Create minimal body data
        # Positions: 2 points * 3 components * 3 bytes = 18 bytes
        positions_data = b'\x00\x00\x00' * 6  # All zeros
        
        # Alphas: 2 points * 1 byte = 2 bytes
        alphas_data = b'\x00\x00'
        
        # Colors: 2 points * 3 components = 6 bytes
        colors_data = b'\x00\x00\x00\x00\x00\x00'
        
        # Scales: 2 points * 3 components = 6 bytes
        scales_data = b'\x00\x00\x00\x00\x00\x00'
        
        # Rotations v2: 2 points * 3 components = 6 bytes
        rotations_data = b'\x00\x00\x00\x00\x00\x00'
        
        body_data = positions_data + alphas_data + colors_data + scales_data + rotations_data
        compressed_body = gzip.compress(body_data)
        
        # Create header bytes
        header_bytes = (
            header.magic.to_bytes(4, 'little') +
            header.version.to_bytes(4, 'little') +
            header.num_points.to_bytes(4, 'little') +
            header.sh_degree.to_bytes(1, 'little') +
            header.fractional_bits.to_bytes(1, 'little') +
            header.flags.to_bytes(1, 'little') +
            header.reserved.to_bytes(1, 'little')
        )
        
        # Create complete SPZ file
        spz_data = header_bytes + compressed_body
        file_obj = io.BytesIO(spz_data)
        
        # Load the data
        result = load(file_obj)
        
        # Validate structure
        expected_keys = {'positions', 'alphas', 'colors', 'scales', 'rotations'}
        assert set(result.keys()) == expected_keys
        
        # Validate shapes
        assert result['positions'].shape == (2, 3)
        assert result['alphas'].shape == (2, 1)
        assert result['colors'].shape == (2, 3)
        assert result['scales'].shape == (2, 3)
        assert result['rotations'].shape == (2, 4)
        
        # Validate data types
        for key in expected_keys:
            assert result[key].dtype == np.float32
        
        # Validate quaternion normalization
        for i in range(2):
            norm = np.linalg.norm(result['rotations'][i])
            assert abs(norm - 1.0) < 1e-6
    
    def test_spz_v3_file(self):
        """Test loading an SPZ v3 file."""
        header = SpzHeader(
            magic=0x5053474E,
            version=3,
            num_points=1,
            sh_degree=0,
            fractional_bits=12,
            flags=0,
            reserved=0
        )
        
        # Create body data for v3 (rotations are 4 bytes each)
        positions_data = b'\x00\x00\x00' * 3  # 1 point * 3 components * 3 bytes
        alphas_data = b'\x00'
        colors_data = b'\x00\x00\x00'
        scales_data = b'\x00\x00\x00'
        rotations_data = b'\x00\x00\x00\x00'  # 1 point * 4 bytes
        
        body_data = positions_data + alphas_data + colors_data + scales_data + rotations_data
        compressed_body = gzip.compress(body_data)
        
        # Create header bytes
        header_bytes = (
            header.magic.to_bytes(4, 'little') +
            header.version.to_bytes(4, 'little') +
            header.num_points.to_bytes(4, 'little') +
            header.sh_degree.to_bytes(1, 'little') +
            header.fractional_bits.to_bytes(1, 'little') +
            header.flags.to_bytes(1, 'little') +
            header.reserved.to_bytes(1, 'little')
        )
        
        spz_data = header_bytes + compressed_body
        file_obj = io.BytesIO(spz_data)
        
        result = load(file_obj)
        
        # Validate that we got the expected data
        assert len(result['positions']) == 1
        assert result['rotations'].shape == (1, 4)
    
    def test_spz_with_sh_coeffs(self):
        """Test loading an SPZ file with spherical harmonics."""
        header = SpzHeader(
            magic=0x5053474E,
            version=2,
            num_points=1,
            sh_degree=1,  # 9 coefficients
            fractional_bits=8,
            flags=0,
            reserved=0
        )
        
        # Create body data including SH coefficients
        positions_data = b'\x00\x00\x00' * 3
        alphas_data = b'\x00'
        colors_data = b'\x00\x00\x00'
        scales_data = b'\x00\x00\x00'
        rotations_data = b'\x00\x00\x00'
        sh_data = b'\x00' * 9  # 9 coefficients
        
        body_data = positions_data + alphas_data + colors_data + scales_data + rotations_data + sh_data
        compressed_body = gzip.compress(body_data)
        
        # Create header bytes
        header_bytes = (
            header.magic.to_bytes(4, 'little') +
            header.version.to_bytes(4, 'little') +
            header.num_points.to_bytes(4, 'little') +
            header.sh_degree.to_bytes(1, 'little') +
            header.fractional_bits.to_bytes(1, 'little') +
            header.flags.to_bytes(1, 'little') +
            header.reserved.to_bytes(1, 'little')
        )
        
        spz_data = header_bytes + compressed_body
        file_obj = io.BytesIO(spz_data)
        
        result = load(file_obj)
        
        # Validate that SH coefficients are present
        assert 'sh_coeffs' in result
        assert result['sh_coeffs'].shape == (1, 3, 3)  # 9 coefficients / 3 colors = 3 per color
    
    def test_spz_without_sh_coeffs(self):
        """Test loading an SPZ file without spherical harmonics."""
        header = SpzHeader(
            magic=0x5053474E,
            version=2,
            num_points=1,
            sh_degree=0,  # No SH coefficients
            fractional_bits=8,
            flags=0,
            reserved=0
        )
        
        # Create body data without SH coefficients
        positions_data = b'\x00\x00\x00' * 3
        alphas_data = b'\x00'
        colors_data = b'\x00\x00\x00'
        scales_data = b'\x00\x00\x00'
        rotations_data = b'\x00\x00\x00'
        
        body_data = positions_data + alphas_data + colors_data + scales_data + rotations_data
        compressed_body = gzip.compress(body_data)
        
        # Create header bytes
        header_bytes = (
            header.magic.to_bytes(4, 'little') +
            header.version.to_bytes(4, 'little') +
            header.num_points.to_bytes(4, 'little') +
            header.sh_degree.to_bytes(1, 'little') +
            header.fractional_bits.to_bytes(1, 'little') +
            header.flags.to_bytes(1, 'little') +
            header.reserved.to_bytes(1, 'little')
        )
        
        spz_data = header_bytes + compressed_body
        file_obj = io.BytesIO(spz_data)
        
        result = load(file_obj)
        
        # Validate that SH coefficients are not present
        assert 'sh_coeffs' not in result
    
    def test_insufficient_body_data(self):
        """Test error handling for insufficient body data."""
        header = SpzHeader(
            magic=0x5053474E,
            version=2,
            num_points=2,
            sh_degree=0,
            fractional_bits=8,
            flags=0,
            reserved=0
        )
        
        # Create incomplete body data
        positions_data = b'\x00\x00\x00' * 3  # Only 1 point instead of 2
        body_data = positions_data
        compressed_body = gzip.compress(body_data)
        
        # Create header bytes
        header_bytes = (
            header.magic.to_bytes(4, 'little') +
            header.version.to_bytes(4, 'little') +
            header.num_points.to_bytes(4, 'little') +
            header.sh_degree.to_bytes(1, 'little') +
            header.fractional_bits.to_bytes(1, 'little') +
            header.flags.to_bytes(1, 'little') +
            header.reserved.to_bytes(1, 'little')
        )
        
        spz_data = header_bytes + compressed_body
        file_obj = io.BytesIO(spz_data)
        
        with pytest.raises(RuntimeError, match="Error reading SPZ file"):
            load(file_obj)
    
    def test_antialiasing_flag(self):
        """Test loading a file with antialiasing flag set."""
        header = SpzHeader(
            magic=0x5053474E,
            version=2,
            num_points=1,
            sh_degree=0,
            fractional_bits=8,
            flags=0x1,  # Antialiasing flag
            reserved=0
        )
        
        # Create minimal body data
        positions_data = b'\x00\x00\x00' * 3
        alphas_data = b'\x00'
        colors_data = b'\x00\x00\x00'
        scales_data = b'\x00\x00\x00'
        rotations_data = b'\x00\x00\x00'
        
        body_data = positions_data + alphas_data + colors_data + scales_data + rotations_data
        compressed_body = gzip.compress(body_data)
        
        # Create header bytes
        header_bytes = (
            header.magic.to_bytes(4, 'little') +
            header.version.to_bytes(4, 'little') +
            header.num_points.to_bytes(4, 'little') +
            header.sh_degree.to_bytes(1, 'little') +
            header.fractional_bits.to_bytes(1, 'little') +
            header.flags.to_bytes(1, 'little') +
            header.reserved.to_bytes(1, 'little')
        )
        
        spz_data = header_bytes + compressed_body
        file_obj = io.BytesIO(spz_data)
        
        result = load(file_obj)
        
        # Should load successfully
        assert 'positions' in result
        assert len(result['positions']) == 1
