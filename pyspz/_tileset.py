"""
3D Tiles tileset generation for SPZ data.

This module provides functionality to convert SPZ Gaussian Splatting data
into 3D Tiles format for efficient spatial visualization and streaming.
"""

import json
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from ._core import load


class TilesetGenerator:
    """Generator for 3D Tiles tilesets from SPZ data."""
    
    def __init__(self, max_points_per_tile: int = 10000, tile_size: float = 1000.0):
        """Initialize tileset generator.
        
        Args:
            max_points_per_tile: Maximum number of Gaussians per tile
            tile_size: Size of each tile in world units
        """
        self.max_points_per_tile = max_points_per_tile
        self.tile_size = tile_size
    
    def generate_tileset(self, spz_file: str, output_dir: str) -> Dict:
        """Generate 3D Tiles tileset from SPZ file.
        
        Args:
            spz_file: Path to SPZ file
            output_dir: Output directory for tileset files
            
        Returns:
            Dictionary containing tileset metadata
        """
        # Load SPZ data
        data = load(spz_file)
        positions = data['positions']
        
        # Calculate bounding box
        bbox = self._calculate_bounding_box(positions)
        
        # Create spatial tiles
        tiles = self._create_spatial_tiles(data, bbox)
        
        # Generate tileset.json
        tileset = self._create_tileset_json(tiles, bbox)
        
        # Save tileset files
        self._save_tileset_files(tileset, tiles, output_dir)
        
        return tileset
    
    def _calculate_bounding_box(self, positions: np.ndarray) -> Dict:
        """Calculate bounding box for positions.
        
        Args:
            positions: (N, 3) array of positions
            
        Returns:
            Dictionary with bounding box information
        """
        if len(positions) == 0:
            return {
                'min': [0, 0, 0],
                'max': [0, 0, 0],
                'center': [0, 0, 0],
                'size': [0, 0, 0]
            }
        
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        center = (min_pos + max_pos) / 2
        size = max_pos - min_pos
        
        return {
            'min': min_pos.tolist(),
            'max': max_pos.tolist(),
            'center': center.tolist(),
            'size': size.tolist()
        }
    
    def _create_spatial_tiles(self, data: Dict[str, np.ndarray], bbox: Dict) -> List[Dict]:
        """Create spatial tiles from Gaussian data.
        
        Args:
            data: SPZ data dictionary
            bbox: Bounding box information
            
        Returns:
            List of tile dictionaries
        """
        positions = data['positions']
        num_points = len(positions)
        
        # Calculate number of tiles needed
        num_tiles = math.ceil(num_points / self.max_points_per_tile)
        
        tiles = []
        for i in range(num_tiles):
            start_idx = i * self.max_points_per_tile
            end_idx = min((i + 1) * self.max_points_per_tile, num_points)
            
            # Extract tile data
            tile_data = self._extract_tile_data(data, start_idx, end_idx)
            
            # Calculate tile bounding box
            tile_bbox = self._calculate_tile_bbox(tile_data['positions'])
            
            # Create tile
            tile = {
                'id': f'tile_{i:03d}',
                'bounding_volume': tile_bbox,
                'geometric_error': self._calculate_geometric_error(tile_data),
                'refine': 'ADD' if i < num_tiles - 1 else 'REPLACE',
                'content': {
                    'uri': f'tiles/tile_{i:03d}.json'
                },
                'data': tile_data
            }
            
            tiles.append(tile)
        
        return tiles
    
    def _extract_tile_data(self, data: Dict[str, np.ndarray], start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """Extract data for a specific tile.
        
        Args:
            data: Full SPZ data
            start_idx: Start index for tile
            end_idx: End index for tile
            
        Returns:
            Dictionary with tile data
        """
        tile_data = {}
        for key, array in data.items():
            if key == 'sh_coeffs' and array.shape[1] == 0:
                # Skip empty SH coefficients
                continue
            tile_data[key] = array[start_idx:end_idx]
        
        return tile_data
    
    def _calculate_tile_bbox(self, positions: np.ndarray) -> Dict:
        """Calculate bounding box for tile positions.
        
        Args:
            positions: (N, 3) array of positions
            
        Returns:
            Dictionary with tile bounding box
        """
        if len(positions) == 0:
            return {'box': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]}
        
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        center = (min_pos + max_pos) / 2
        size = max_pos - min_pos
        
        # Create oriented bounding box
        return {
            'box': [
                center[0], center[1], center[2],  # center
                1, 0, 0,  # x-axis
                0, 1, 0,  # y-axis
                0, 0, 1,  # z-axis
                size[0] / 2, size[1] / 2, size[2] / 2  # half-sizes
            ]
        }
    
    def _calculate_geometric_error(self, tile_data: Dict[str, np.ndarray]) -> float:
        """Calculate geometric error for tile.
        
        Args:
            tile_data: Tile data dictionary
            
        Returns:
            Geometric error value
        """
        positions = tile_data['positions']
        if len(positions) == 0:
            return 0.0
        
        # Calculate average distance between points
        if len(positions) < 2:
            return 1.0
        
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, min(i + 10, len(positions))):  # Sample nearby points
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        if not distances:
            return 1.0
        
        return np.mean(distances)
    
    def _create_tileset_json(self, tiles: List[Dict], bbox: Dict) -> Dict:
        """Create tileset.json structure.
        
        Args:
            tiles: List of tile dictionaries
            bbox: Bounding box information
            
        Returns:
            Tileset JSON structure
        """
        # Calculate root bounding box
        root_bbox = {
            'box': [
                bbox['center'][0], bbox['center'][1], bbox['center'][2],  # center
                1, 0, 0,  # x-axis
                0, 1, 0,  # y-axis
                0, 0, 1,  # z-axis
                bbox['size'][0] / 2, bbox['size'][1] / 2, bbox['size'][2] / 2  # half-sizes
            ]
        }
        
        # Calculate root geometric error
        root_geometric_error = max(tile['geometric_error'] for tile in tiles) if tiles else 1.0
        
        tileset = {
            'asset': {
                'version': '1.0',
                'generator': 'PySpz Tileset Generator',
                'generatorVersion': '0.1.0'
            },
            'properties': {
                'numberOfPoints': sum(len(tile['data']['positions']) for tile in tiles),
                'numberOfTiles': len(tiles)
            },
            'geometricError': root_geometric_error,
            'root': {
                'boundingVolume': root_bbox,
                'geometricError': root_geometric_error,
                'refine': 'ADD',
                'children': tiles
            }
        }
        
        return tileset
    
    def _save_tileset_files(self, tileset: Dict, tiles: List[Dict], output_dir: str) -> None:
        """Save tileset files to output directory.
        
        Args:
            tileset: Tileset JSON structure
            tiles: List of tile dictionaries
            output_dir: Output directory path
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tiles'), exist_ok=True)
        
        # Save tileset.json
        tileset_path = os.path.join(output_dir, 'tileset.json')
        with open(tileset_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            tileset_serializable = self._convert_to_serializable(tileset)
            json.dump(tileset_serializable, f, indent=2)
        
        # Save individual tile files
        for tile in tiles:
            tile_path = os.path.join(output_dir, tile['content']['uri'])
            with open(tile_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                tile_data = tile['data'].copy()
                for key, array in tile_data.items():
                    if isinstance(array, np.ndarray):
                        tile_data[key] = array.tolist()
                
                tile_data_serializable = self._convert_to_serializable(tile_data)
                json.dump(tile_data_serializable, f, indent=2)
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Object with numpy types converted to Python types
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def spz_to_tileset(spz_file: str, output_dir: str, max_points_per_tile: int = 10000) -> Dict:
    """Convert SPZ file to 3D Tiles tileset.
    
    Args:
        spz_file: Path to SPZ file
        output_dir: Output directory for tileset files
        max_points_per_tile: Maximum number of Gaussians per tile
        
    Returns:
        Dictionary containing tileset metadata
        
    Example:
        >>> import pyspz
        >>> tileset = pyspz.spz_to_tileset("model.spz", "output/")
        >>> print(f"Created {tileset['properties']['numberOfTiles']} tiles")
    """
    generator = TilesetGenerator(max_points_per_tile=max_points_per_tile)
    return generator.generate_tileset(spz_file, output_dir)
