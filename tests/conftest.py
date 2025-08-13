import pytest
import numpy as np
import zarr
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_zarr_array():
    """Create a temporary zarr array for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple 4D test array (C, Z, Y, X)
    test_data = np.random.randint(0, 255, size=(2, 64, 128, 128), dtype=np.uint8)
    
    store = zarr.DirectoryStore(temp_dir)
    root = zarr.group(store=store)
    
    # Create s0 (base scale)
    arr = root.create_dataset(
        's0',
        data=test_data,
        chunks=(2, 32, 64, 64),
        compressor=zarr.Blosc(),
        dtype=np.uint8
    )
    
    # Add multiscale metadata
    root.attrs['multiscales'] = [{
        'version': '0.4',
        'axes': [
            {'name': 'c', 'type': 'channel'},
            {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'x', 'type': 'space', 'unit': 'micrometer'}
        ],
        'datasets': [{
            'path': 's0',
            'coordinateTransformations': [
                {'type': 'scale', 'scale': [1.0, 1.0, 1.0, 1.0]},
                {'type': 'translation', 'translation': [0.0, 0.0, 0.0, 0.0]}
            ]
        }]
    }]
    
    yield root, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

# @pytest.fixture
# def mock_dask_client():
#     """Create a mock dask client for testing."""
#     from unittest.mock import Mock
    
#     client = Mock()
#     client.map.return_value = [Mock()]
#     return client