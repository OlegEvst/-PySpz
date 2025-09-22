"""
Tests for 3D Tiles tileset generation functionality.
"""

import pytest
import numpy as np
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pyspz._tileset import TilesetGenerator, spz_to_tileset
from pyspz import spz_to_tileset as public_spz_to_tileset


class TestTilesetGenerator:
    """Test cases for TilesetGenerator class."""
    
    def test_init(self):
        """Test TilesetGenerator initialization."""
        generator = TilesetGenerator(max_points_per_tile=5000, tile_size=500.0)
        
        assert generator.max_points_per_tile == 5000
        assert generator.tile_size == 500.0
    
    def test_calculate_bounding_box(self):
        """Test bounding box calculation."""
        generator = TilesetGenerator()
        
        # Test with simple positions
        positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        bbox = generator._calculate_bounding_box(positions)
        
        assert bbox['min'] == [0, 0, 0]
        assert bbox['max'] == [2, 2, 2]
        assert bbox['center'] == [1, 1, 1]
        assert bbox['size'] == [2, 2, 2]
    
    def test_calculate_bounding_box_empty(self):
        """Test bounding box calculation with empty positions."""
        generator = TilesetGenerator()
        
        positions = np.array([], dtype=np.float32).reshape(0, 3)
        bbox = generator._calculate_bounding_box(positions)
        
        assert bbox['min'] == [0, 0, 0]
        assert bbox['max'] == [0, 0, 0]
        assert bbox['center'] == [0, 0, 0]
        assert bbox['size'] == [0, 0, 0]
    
    def test_create_spatial_tiles(self):
        """Test spatial tiles creation."""
        generator = TilesetGenerator(max_points_per_tile=2)
        
        # Create test data
        data = {
            'positions': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32),
            'colors': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32),
            'alphas': np.array([[0.5], [0.6], [0.7], [0.8]], dtype=np.float32),
            'scales': np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float32),
            'rotations': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        }
        
        bbox = {'min': [0, 0, 0], 'max': [3, 3, 3], 'center': [1.5, 1.5, 1.5], 'size': [3, 3, 3]}
        tiles = generator._create_spatial_tiles(data, bbox)
        
        # Should create 2 tiles (4 points / 2 points per tile)
        assert len(tiles) == 2
        
        # Check first tile
        tile1 = tiles[0]
        assert tile1['id'] == 'tile_000'
        assert len(tile1['data']['positions']) == 2
        assert tile1['refine'] == 'ADD'
        assert 'bounding_volume' in tile1
        assert 'geometric_error' in tile1
        
        # Check second tile
        tile2 = tiles[1]
        assert tile2['id'] == 'tile_001'
        assert len(tile2['data']['positions']) == 2
        assert tile2['refine'] == 'REPLACE'  # Last tile should be REPLACE
    
    def test_extract_tile_data(self):
        """Test tile data extraction."""
        generator = TilesetGenerator()
        
        data = {
            'positions': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32),
            'colors': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            'sh_coeffs': np.array([], dtype=np.float32).reshape(0, 0, 3)  # Empty SH
        }
        
        tile_data = generator._extract_tile_data(data, 1, 3)
        
        assert len(tile_data['positions']) == 2
        assert len(tile_data['colors']) == 2
        assert 'sh_coeffs' not in tile_data  # Should be skipped if empty
    
    def test_calculate_tile_bbox(self):
        """Test tile bounding box calculation."""
        generator = TilesetGenerator()
        
        positions = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)
        bbox = generator._calculate_tile_bbox(positions)
        
        assert 'box' in bbox
        assert len(bbox['box']) == 15  # 3D box has 15 components (center + 3 axes + 3 half-sizes)
    
    def test_calculate_tile_bbox_empty(self):
        """Test tile bounding box calculation with empty positions."""
        generator = TilesetGenerator()
        
        positions = np.array([], dtype=np.float32).reshape(0, 3)
        bbox = generator._calculate_tile_bbox(positions)
        
        assert 'box' in bbox
        assert bbox['box'] == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    
    def test_calculate_geometric_error(self):
        """Test geometric error calculation."""
        generator = TilesetGenerator()
        
        # Test with points at known distances
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        tile_data = {'positions': positions}
        
        error = generator._calculate_geometric_error(tile_data)
        assert error > 0
        
        # Test with single point
        single_pos = np.array([[0, 0, 0]], dtype=np.float32)
        single_data = {'positions': single_pos}
        
        error = generator._calculate_geometric_error(single_data)
        assert error == 1.0
    
    def test_create_tileset_json(self):
        """Test tileset JSON creation."""
        generator = TilesetGenerator()
        
        # Create mock tiles
        tiles = [
            {
                'id': 'tile_000',
                'bounding_volume': {'box': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
                'geometric_error': 1.0,
                'refine': 'ADD',
                'content': {'uri': 'tiles/tile_000.json'},
                'data': {'positions': np.array([[0, 0, 0]], dtype=np.float32)}
            }
        ]
        
        bbox = {'center': [0, 0, 0], 'size': [2, 2, 2]}
        tileset = generator._create_tileset_json(tiles, bbox)
        
        assert 'asset' in tileset
        assert 'properties' in tileset
        assert 'geometricError' in tileset
        assert 'root' in tileset
        
        assert tileset['asset']['generator'] == 'PySpz Tileset Generator'
        assert tileset['properties']['numberOfTiles'] == 1
        assert tileset['root']['children'] == tiles


class TestSpzToTileset:
    """Test cases for spz_to_tileset function."""
    
    def test_spz_to_tileset_function(self):
        """Test spz_to_tileset function."""
        # This is a basic test - in real usage, you'd need actual SPZ files
        with patch('pyspz._tileset.load') as mock_load:
            mock_load.return_value = {
                'positions': np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
                'colors': np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
                'alphas': np.array([[0.5], [0.6]], dtype=np.float32),
                'scales': np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
                'rotations': np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            }
            
            result = spz_to_tileset('test.spz', 'output/')
            
            mock_load.assert_called_once_with('test.spz')
            assert 'asset' in result
            assert 'properties' in result
    
    def test_public_api(self):
        """Test public API function."""
        with patch('pyspz._tileset.spz_to_tileset') as mock_func:
            mock_func.return_value = {'test': 'result'}
            
            result = public_spz_to_tileset('test.spz', 'output/')
            
            mock_func.assert_called_once_with('test.spz', 'output/', 10000)
            assert result == {'test': 'result'}


class TestTilesetIntegration:
    """Integration tests for tileset functionality."""
    
    def test_save_tileset_files(self):
        """Test saving tileset files to disk."""
        generator = TilesetGenerator()
        
        # Create mock tileset and tiles
        tileset = {
            'asset': {'version': '1.0'},
            'properties': {'numberOfPoints': 2, 'numberOfTiles': 1},
            'geometricError': 1.0,
            'root': {'children': []}
        }
        
        tiles = [
            {
                'id': 'tile_000',
                'bounding_volume': {'box': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
                'geometric_error': 1.0,
                'refine': 'REPLACE',
                'content': {'uri': 'tiles/tile_000.json'},
                'data': {
                    'positions': np.array([[0, 0, 0]], dtype=np.float32),
                    'colors': np.array([[1, 0, 0]], dtype=np.float32)
                }
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator._save_tileset_files(tileset, tiles, temp_dir)
            
            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, 'tileset.json'))
            assert os.path.exists(os.path.join(temp_dir, 'tiles', 'tile_000.json'))
            
            # Check tileset.json content
            with open(os.path.join(temp_dir, 'tileset.json'), 'r') as f:
                saved_tileset = json.load(f)
                assert saved_tileset['asset']['version'] == '1.0'
            
            # Check tile file content
            with open(os.path.join(temp_dir, 'tiles', 'tile_000.json'), 'r') as f:
                saved_tile = json.load(f)
                assert 'positions' in saved_tile
                assert 'colors' in saved_tile
    
    def test_full_tileset_generation(self):
        """Test complete tileset generation process."""
        with patch('pyspz._tileset.load') as mock_load:
            # Create realistic test data
            mock_load.return_value = {
                'positions': np.array([
                    [0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]
                ], dtype=np.float32),
                'colors': np.array([
                    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]
                ], dtype=np.float32),
                'alphas': np.array([[0.5], [0.6], [0.7], [0.8], [0.9]], dtype=np.float32),
                'scales': np.array([
                    [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]
                ], dtype=np.float32),
                'rotations': np.array([
                    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]
                ], dtype=np.float32)
            }
            
            generator = TilesetGenerator(max_points_per_tile=2)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = generator.generate_tileset('test.spz', temp_dir)
                
                # Check result structure
                assert 'asset' in result
                assert 'properties' in result
                assert 'geometricError' in result
                assert 'root' in result
                
                # Check that files were created
                assert os.path.exists(os.path.join(temp_dir, 'tileset.json'))
                
                # Should create 3 tiles (5 points / 2 points per tile = 3 tiles)
                assert result['properties']['numberOfTiles'] == 3
                assert result['properties']['numberOfPoints'] == 5
