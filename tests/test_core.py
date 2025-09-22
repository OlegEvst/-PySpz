"""
Tests for core SPZ loading functionality.
"""

import pytest
import numpy as np
import gzip
import io
from unittest.mock import patch, mock_open
from pyspz._core import load, _load_from_stream, _parse_body, _validate_quaternions, _validate_data_consistency
from pyspz._header import SpzHeader


class TestCoreLoading:
    """Test cases for core loading functionality."""
    
    def test_load_from_file_path(self):
        """Test loading from file path."""
        # Create a mock SPZ file
        header_data = b'\x4E\x47\x53\x50' + b'\x02\x00\x00\x00' + b'\x02\x00\x00\x00' + b'\x00\x08\x00\x00'
        body_data = b'test_data'
        compressed_body = gzip.compress(body_data)
        
        with patch('builtins.open', mock_open(read_data=header_data + compressed_body)):
            with patch('pyspz._core._load_from_stream') as mock_load:
                mock_load.return_value = {'positions': np.zeros((2, 3), dtype=np.float32)}
                result = load('test.spz')
                mock_load.assert_called_once()
    
    def test_load_from_file_object(self):
        """Test loading from file-like object."""
        header_data = b'\x4E\x47\x53\x50' + b'\x02\x00\x00\x00' + b'\x02\x00\x00\x00' + b'\x00\x08\x00\x00'
        body_data = b'test_data'
        compressed_body = gzip.compress(body_data)
        
        file_obj = io.BytesIO(header_data + compressed_body)
        
        with patch('pyspz._core._load_from_stream') as mock_load:
            mock_load.return_value = {'positions': np.zeros((2, 3), dtype=np.float32)}
            result = load(file_obj)
            mock_load.assert_called_once()
    
    def test_empty_file(self):
        """Test handling of empty files."""
        header = SpzHeader(
            magic=0x5053474E,
            version=2,
            num_points=0,
            sh_degree=0,
            fractional_bits=8,
            flags=0,
            reserved=0
        )
        
        result = _parse_body(b'', header)
        
        expected = {
            'positions': np.zeros((0, 3), dtype=np.float32),
            'alphas': np.zeros((0, 1), dtype=np.float32),
            'colors': np.zeros((0, 3), dtype=np.float32),
            'scales': np.zeros((0, 3), dtype=np.float32),
            'rotations': np.zeros((0, 4), dtype=np.float32),
        }
        
        for key in expected:
            assert key in result
            assert result[key].shape == expected[key].shape
            assert result[key].dtype == expected[key].dtype
    
    def test_insufficient_header_data(self):
        """Test error handling for insufficient header data."""
        short_data = b'short'
        file_obj = io.BytesIO(short_data)
        
        with pytest.raises(RuntimeError, match="Error reading SPZ file"):
            _load_from_stream(file_obj)
    
    def test_no_compressed_data(self):
        """Test error handling for missing compressed data."""
        header_data = b'\x4E\x47\x53\x50' + b'\x02\x00\x00\x00' + b'\x02\x00\x00\x00' + b'\x00\x08\x00\x00'
        file_obj = io.BytesIO(header_data)
        
        with pytest.raises(RuntimeError, match="Error reading SPZ file"):
            _load_from_stream(file_obj)
    
    def test_invalid_gzip_data(self):
        """Test error handling for invalid gzip data."""
        header_data = b'\x4E\x47\x53\x50' + b'\x02\x00\x00\x00' + b'\x02\x00\x00\x00' + b'\x00\x08\x00\x00'
        invalid_gzip = b'invalid_gzip_data'
        file_obj = io.BytesIO(header_data + invalid_gzip)
        
        with pytest.raises(RuntimeError, match="Error reading SPZ file"):
            _load_from_stream(file_obj)


class TestQuaternionValidation:
    """Test cases for quaternion validation."""
    
    def test_valid_quaternions(self):
        """Test validation of properly normalized quaternions."""
        quaternions = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        # Normalize to unit length
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        
        # Should not raise an exception
        _validate_quaternions(quaternions)
    
    def test_invalid_quaternions(self):
        """Test validation of improperly normalized quaternions."""
        quaternions = np.array([[1.0, 0.0, 0.0, 0.0],
                               [2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Quaternions are not properly normalized"):
            _validate_quaternions(quaternions)


class TestDataConsistency:
    """Test cases for data consistency validation."""
    
    def test_consistent_data(self):
        """Test validation of consistent data arrays."""
        data = {
            'positions': np.zeros((10, 3), dtype=np.float32),
            'alphas': np.zeros((10, 1), dtype=np.float32),
            'colors': np.zeros((10, 3), dtype=np.float32),
            'scales': np.zeros((10, 3), dtype=np.float32),
            'rotations': np.zeros((10, 4), dtype=np.float32),
        }
        
        # Should not raise an exception
        _validate_data_consistency(data)
    
    def test_inconsistent_data(self):
        """Test validation of inconsistent data arrays."""
        data = {
            'positions': np.zeros((10, 3), dtype=np.float32),
            'alphas': np.zeros((5, 1), dtype=np.float32),  # Wrong size
        }
        
        with pytest.raises(ValueError, match="Array 'alphas' has 5 points, expected 10"):
            _validate_data_consistency(data)
    
    def test_empty_data(self):
        """Test validation of empty data."""
        data = {}
        
        # Should not raise an exception
        _validate_data_consistency(data)
    
    def test_sh_coeffs_validation(self):
        """Test validation with spherical harmonics coefficients."""
        data = {
            'positions': np.zeros((10, 3), dtype=np.float32),
            'sh_coeffs': np.zeros((10, 9, 3), dtype=np.float32),
        }
        
        # Should not raise an exception
        _validate_data_consistency(data)


class TestErrorHandling:
    """Test cases for error handling in core functionality."""
    
    def test_runtime_error_wrapping(self):
        """Test that exceptions are properly wrapped as RuntimeError."""
        with patch('pyspz._core.SpzHeader.from_bytes') as mock_header:
            mock_header.side_effect = Exception("Test error")
            
            file_obj = io.BytesIO(b'test_data')
            
            with pytest.raises(RuntimeError, match="Error reading SPZ file"):
                _load_from_stream(file_obj)
