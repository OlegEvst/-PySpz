# PySpz

A Python library for reading SPZ (3D Gaussian Splatting) files. This library provides functionality to load and decode SPZ files, which are compressed 3D Gaussian Splatting data files.

## Features

- **Full SPZ Format Support**: Supports SPZ versions 2 and 3
- **Spherical Harmonics**: Handles spherical harmonics coefficients (degrees 0-3)
- **3D Tiles Integration**: Convert SPZ files to 3D Tiles tilesets for spatial visualization
- **Vectorized Operations**: Efficient NumPy-based decoding
- **Minimal Dependencies**: Only requires NumPy (SciPy optional for quaternion operations)
- **Binary Compatibility**: Maintains compatibility with the original C++ implementation

## Installation

```bash
pip install pyspz
```

## Quick Start

```python
import pyspz
import numpy as np

# Load SPZ file
data = pyspz.load("model.spz")

# Access Gaussian attributes
positions = data['positions']  # (N, 3) XYZ coordinates
alphas = data['alphas']       # (N, 1) opacity values
colors = data['colors']       # (N, 3) RGB colors [0,1]
scales = data['scales']       # (N, 3) XYZ scales
rotations = data['rotations'] # (N, 4) quaternions (w,x,y,z)

# Spherical harmonics (if present)
if 'sh_coeffs' in data:
    sh_coeffs = data['sh_coeffs']  # (N, K, 3) SH coefficients
```

### 3D Tiles Tileset Generation

```python
import pyspz

# Convert SPZ file to 3D Tiles tileset
tileset = pyspz.spz_to_tileset("model.spz", "output/", max_points_per_tile=10000)

print(f"Created {tileset['properties']['numberOfTiles']} tiles")
print(f"Total points: {tileset['properties']['numberOfPoints']}")

# The tileset is saved as:
# - output/tileset.json (main tileset file)
# - output/tiles/tile_000.json (individual tile files)
# - output/tiles/tile_001.json
# - ...
```

## Supported SPZ Versions

- **Version 2**: Uses 3×8-bit signed quaternions
- **Version 3**: Uses compressed 4-byte quaternions

## Data Format

The library returns a dictionary containing NumPy arrays:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `positions` | (N, 3) | float32 | XYZ coordinates |
| `alphas` | (N, 1) | float32 | Opacity values |
| `colors` | (N, 3) | float32 | RGB colors [0,1] |
| `scales` | (N, 3) | float32 | XYZ scales |
| `rotations` | (N, 4) | float32 | Quaternions (w,x,y,z) |
| `sh_coeffs` | (N, K, 3) | float32 | Spherical harmonics (optional) |

Where:
- `N` = number of Gaussian points
- `K` = number of SH coefficients per color (depends on degree)

## Spherical Harmonics

The library supports spherical harmonics with degrees 0-3:

| Degree | Coefficients per Color | Total per Point |
|--------|----------------------|-----------------|
| 0 | 0 | 0 |
| 1 | 3 | 9 |
| 2 | 8 | 24 |
| 3 | 15 | 45 |

## Performance

PySpz is optimized for performance with large datasets:

- **Vectorized Operations**: All decoding uses NumPy vectorization
- **Memory Efficient**: Streaming decompression without full file loading
- **Fast Quaternion Operations**: Optimized quaternion normalization

### Benchmark

On a modern machine (2023), loading a 1M point SPZ file typically takes:
- **Loading**: ~2 seconds
- **Memory Usage**: ~200MB peak

## API Reference

### `pyspz.load(source)`

Load 3D Gaussian Splatting data from an SPZ file.

**Parameters:**
- `source` (str or BinaryIO): Path to .spz file or binary file-like object

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary containing Gaussian attributes

**Raises:**
- `ValueError`: If the file format is invalid or unsupported
- `RuntimeError`: If there are issues reading the file

### `pyspz.spz_to_tileset(spz_file, output_dir, max_points_per_tile=10000)`

Convert SPZ file to 3D Tiles tileset for spatial visualization.

**Parameters:**
- `spz_file` (str): Path to .spz file
- `output_dir` (str): Output directory for tileset files
- `max_points_per_tile` (int): Maximum number of Gaussians per tile (default: 10000)

**Returns:**
- `Dict`: Dictionary containing tileset metadata

**Raises:**
- `ValueError`: If the SPZ file is invalid
- `RuntimeError`: If there are issues creating the tileset

## Requirements

- Python 3.8+
- NumPy 1.20.0+
- SciPy 1.7.0+ (optional, recommended for quaternion operations)

## Development

### Setup

```bash
git clone https://github.com/OlegEvst/-PySpz.git
cd -PySpz
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black pyspz/ tests/
flake8 pyspz/ tests/
mypy pyspz/
```

## License

MIT License. See LICENSE file for details.

## Acknowledgments

- Original C++ implementation: [nianticlabs/spz](https://github.com/nianticlabs/spz)
- Java port: [javagl/JSpz](https://github.com/javagl/JSpz)
- This Python port maintains binary compatibility with both implementations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### 0.1.0
- Initial release
- Support for SPZ versions 2 and 3
- Spherical harmonics support (degrees 0-3)
- Vectorized decoding operations
- 3D Tiles tileset generation
- Comprehensive test suite
