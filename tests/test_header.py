"""
Tests for SPZ header parsing.
"""

import pytest
import struct
from pyspz._header import SpzHeader


class TestSpzHeader:
    """Test cases for SpzHeader class."""
    
    def test_valid_header_v2(self):
        """Test parsing a valid SPZ v2 header."""
        # Create a valid header
        magic = 0x5053474E  # 'NGSP'
        version = 2
        num_points = 1000
        sh_degree = 1
        fractional_bits = 8
        flags = 0x1  # antialiasing
        reserved = 0
        
        header_bytes = struct.pack("<I I I B B B B", 
                                 magic, version, num_points, sh_degree, 
                                 fractional_bits, flags, reserved)
        
        header = SpzHeader.from_bytes(header_bytes)
        
        assert header.magic == magic
        assert header.version == version
        assert header.num_points == num_points
        assert header.sh_degree == sh_degree
        assert header.fractional_bits == fractional_bits
        assert header.flags == flags
        assert header.reserved == reserved
        assert header.get_sh_coeff_count() == 9
        assert header.has_antialiasing() == True
    
    def test_valid_header_v3(self):
        """Test parsing a valid SPZ v3 header."""
        magic = 0x5053474E
        version = 3
        num_points = 5000
        sh_degree = 2
        fractional_bits = 12
        flags = 0x0
        reserved = 0
        
        header_bytes = struct.pack("<I I I B B B B", 
                                 magic, version, num_points, sh_degree, 
                                 fractional_bits, flags, reserved)
        
        header = SpzHeader.from_bytes(header_bytes)
        
        assert header.version == version
        assert header.num_points == num_points
        assert header.sh_degree == sh_degree
        assert header.get_sh_coeff_count() == 24
        assert header.has_antialiasing() == False
    
    def test_invalid_magic(self):
        """Test error handling for invalid magic number."""
        invalid_magic = 0x12345678
        header_bytes = struct.pack("<I I I B B B B",
                                 invalid_magic, 2, 100, 0, 8, 0, 0)
        
        with pytest.raises(ValueError, match="Invalid magic number"):
            SpzHeader.from_bytes(header_bytes)
    
    def test_unsupported_version(self):
        """Test error handling for unsupported version."""
        header_bytes = struct.pack("<I I I B B B B",
                                 0x5053474E, 1, 100, 0, 8, 0, 0)
        
        with pytest.raises(ValueError, match="Unsupported SPZ version"):
            SpzHeader.from_bytes(header_bytes)
    
    def test_invalid_sh_degree(self):
        """Test error handling for invalid sh_degree."""
        header_bytes = struct.pack("<I I I B B B B",
                                 0x5053474E, 2, 100, 4, 8, 0, 0)
        
        with pytest.raises(ValueError, match="Invalid sh_degree"):
            SpzHeader.from_bytes(header_bytes)
    
    def test_invalid_fractional_bits(self):
        """Test error handling for invalid fractional_bits."""
        header_bytes = struct.pack("<I I I B B B B",
                                 0x5053474E, 2, 100, 0, 25, 0, 0)
        
        with pytest.raises(ValueError, match="Invalid fractional_bits"):
            SpzHeader.from_bytes(header_bytes)
    
    def test_nonzero_reserved(self):
        """Test error handling for non-zero reserved field."""
        header_bytes = struct.pack("<I I I B B B B",
                                 0x5053474E, 2, 100, 0, 8, 0, 1)
        
        with pytest.raises(ValueError, match="Reserved field must be 0"):
            SpzHeader.from_bytes(header_bytes)
    
    def test_wrong_header_size(self):
        """Test error handling for wrong header size."""
        short_header = b"short"
        
        with pytest.raises(ValueError, match="Header must be exactly 16 bytes"):
            SpzHeader.from_bytes(short_header)
    
    def test_sh_coeff_counts(self):
        """Test spherical harmonics coefficient counts."""
        for sh_degree, expected_count in [(0, 0), (1, 9), (2, 24), (3, 45)]:
            header_bytes = struct.pack("<I I I B B B B",
                                     0x5053474E, 2, 100, sh_degree, 8, 0, 0)
            header = SpzHeader.from_bytes(header_bytes)
            assert header.get_sh_coeff_count() == expected_count
    
    def test_string_representation(self):
        """Test string representation of header."""
        header_bytes = struct.pack("<I I I B B B B",
                                 0x5053474E, 2, 1000, 1, 8, 0x1, 0)
        header = SpzHeader.from_bytes(header_bytes)
        
        str_repr = str(header)
        assert "version=2" in str_repr
        assert "points=1000" in str_repr
        assert "sh_degree=1" in str_repr
