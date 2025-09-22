"""
Header parsing for SPZ files.

This module handles the 16-byte header structure of SPZ files,
including validation and data extraction.
"""

from dataclasses import dataclass
from typing import Tuple
import struct


@dataclass
class SpzHeader:
    """SPZ file header containing metadata about the Gaussian data.
    
    Attributes:
        magic: Magic number, should be 0x5053474E ('NGSP')
        version: File format version (2 or 3)
        num_points: Number of Gaussian points
        sh_degree: Degree of spherical harmonics (0-3)
        fractional_bits: Number of fractional bits for 24-bit positions
        flags: Bit field flags (0x1 = antialiasing)
        reserved: Reserved field, should be 0
    """
    magic: int
    version: int
    num_points: int
    sh_degree: int
    fractional_bits: int
    flags: int
    reserved: int
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'SpzHeader':
        """Parse header from 16-byte binary data.
        
        Args:
            data: Exactly 16 bytes of header data
            
        Returns:
            Parsed SpzHeader object
            
        Raises:
            ValueError: If data is not 16 bytes or contains invalid values
        """
        if len(data) != 16:
            raise ValueError(f"Header must be exactly 16 bytes, got {len(data)}")
        
        # Unpack header using little-endian format
        # Format: <I I I B B B B
        # I = uint32, B = uint8
        magic, version, num_points, sh_degree, fractional_bits, flags, reserved = struct.unpack(
            "<I I I B B B B", data
        )
        
        # Validate magic number
        if magic != 0x5053474E:  # 'NGSP' in ASCII
            raise ValueError(f"Invalid magic number: 0x{magic:08X}, expected 0x5053474E")
        
        # Validate version
        if version not in (2, 3):
            raise ValueError(f"Unsupported SPZ version: {version}, supported versions: 2, 3")
        
        # Validate sh_degree
        if not 0 <= sh_degree <= 3:
            raise ValueError(f"Invalid sh_degree: {sh_degree}, must be 0-3")
        
        # Validate fractional_bits
        if not 0 <= fractional_bits <= 24:
            raise ValueError(f"Invalid fractional_bits: {fractional_bits}, must be 0-24")
        
        # Validate reserved field
        if reserved != 0:
            raise ValueError(f"Reserved field must be 0, got {reserved}")
        
        return cls(
            magic=magic,
            version=version,
            num_points=num_points,
            sh_degree=sh_degree,
            fractional_bits=fractional_bits,
            flags=flags,
            reserved=reserved
        )
    
    def get_sh_coeff_count(self) -> int:
        """Get the number of spherical harmonics coefficients per point.
        
        Returns:
            Number of SH coefficients (0, 9, 24, or 45)
        """
        sh_counts = {0: 0, 1: 9, 2: 24, 3: 45}
        return sh_counts[self.sh_degree]
    
    def has_antialiasing(self) -> bool:
        """Check if the model was trained with antialiasing.
        
        Returns:
            True if antialiasing flag is set
        """
        return bool(self.flags & 0x1)
    
    def __str__(self) -> str:
        """String representation of the header."""
        return (f"SpzHeader(version={self.version}, points={self.num_points}, "
                f"sh_degree={self.sh_degree}, fractional_bits={self.fractional_bits}, "
                f"flags=0x{self.flags:02X})")
