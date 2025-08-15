import pytest
from unittest.mock import patch
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dask_utils import initialize_dask_client
from to_multiscale import (
    upscale_slice, 
    downsample_save_chunk_mode,
    create_multiscale,
    get_downsampling_factors
)


class TestUpscaleSlice:
    """Test the upscale_slice function."""
    
    def test_upscale_slice_basic(self):
        """Test basic upscale functionality."""
        slc = slice(0, 10, None)
        result = upscale_slice(slc, 2)
        assert result == slice(0, 20, None)
    
    def test_upscale_slice_with_start(self):
        """Test upscale with non-zero start."""
        slc = slice(5, 15, None)
        result = upscale_slice(slc, 2)
        assert result == slice(10, 30, None)
    
    def test_upscale_slice_factor_one(self):
        """Test upscale with factor 1 (no change)."""
        slc = slice(0, 10, None)
        result = upscale_slice(slc, 1)
        assert result == slice(0, 10, None)
        
class TestDownsampleSaveChunkMode:
    """Test the downsample_save_chunk_mode function."""
    
    def test_downsample_segmentation(self, temp_zarr_array):
        """Test downsampling segmentation data."""
        z_root, temp_dir = temp_zarr_array
        source_arr = z_root['s0']
        axes_order = [axis['name'].lower() for axis in z_root.attrs['multiscales'][0]['axes']]

        # Get downsampling factors first
        downsampling_factors = tuple(get_downsampling_factors(source_arr.shape, axes_order, high_aspect_ratio=True))
        
        # Create destination array with proper shape calculation
        dest_shape = [dim // factor for dim, factor in zip(source_arr.shape, downsampling_factors)]
        dest_arr = z_root.create_dataset(
            's1',
            shape=dest_shape,
            chunks=source_arr.chunks,
            dtype=source_arr.dtype
        )
        
        # Use output slices that match the actual destination shape
        out_slices = (slice(0, 1), slice(0, dest_shape[1]), slice(0, dest_shape[2]), slice(0, dest_shape[3]))
        result = downsample_save_chunk_mode(
            source_arr, dest_arr, out_slices, 
            downsampling_factors, 'segmentations', False
        )
        
        assert result == 0
        assert dest_arr[out_slices].shape == (1, dest_shape[1], dest_shape[2], dest_shape[3])
    
    def test_downsample_raw_no_antialiasing(self, temp_zarr_array):
        """Test downsampling raw data without antialiasing."""
        z_root, temp_dir = temp_zarr_array
        source_arr = z_root['s0']
        axes_order = [axis['name'].lower() for axis in z_root.attrs['multiscales'][0]['axes']]
        
        # Get downsampling factors that work with the actual array shape
        downsampling_factors = get_downsampling_factors(source_arr.shape, axes_order, high_aspect_ratio=True)
        
        dest_shape = [dim // factor for dim, factor in zip(source_arr.shape, downsampling_factors)]
        dest_arr = z_root.create_dataset(
            's1',
            shape=dest_shape,
            chunks=source_arr.chunks,
            dtype=source_arr.dtype
        )
        
        out_slices = (slice(0, 1), slice(0, dest_shape[1]), slice(0, dest_shape[2]), slice(0, dest_shape[3]))
        
        result = downsample_save_chunk_mode(
            source_arr, dest_arr, out_slices,
            downsampling_factors, 'raw', False
        )
        
        assert result == 0
    
    @patch('src.to_multiscale.ndi.gaussian_filter')
    def test_downsample_raw_with_antialiasing(self, mock_filter, temp_zarr_array):
        """Test downsampling raw data with antialiasing."""
        z_root, temp_dir = temp_zarr_array
        source_arr = z_root['s0']
        axes_order = [axis['name'].lower() for axis in z_root.attrs['multiscales'][0]['axes']]
        
        # Get downsampling factors that work with the actual array shape
        downsampling_factors = get_downsampling_factors(source_arr.shape, axes_order, high_aspect_ratio=True)
        
        dest_shape = [dim // factor for dim, factor in zip(source_arr.shape, downsampling_factors)]
        dest_arr = z_root.create_dataset(
            's1',
            shape=dest_shape,
            chunks=source_arr.chunks,
            dtype=source_arr.dtype
        )
        
        # Mock the gaussian filter to return properly sized data
        expected_in_shape = tuple(
            slice_obj.stop - slice_obj.start if isinstance(slice_obj, slice) else 1
            for slice_obj in (slice(0, 1), slice(0, dest_shape[1] * downsampling_factors[1]), 
                             slice(0, dest_shape[2] * downsampling_factors[2]), 
                             slice(0, dest_shape[3] * downsampling_factors[3]))
        )
        mock_filter.return_value = np.zeros(expected_in_shape, dtype=source_arr.dtype)
        
        out_slices = (slice(0, 1), slice(0, dest_shape[1]), slice(0, dest_shape[2]), slice(0, dest_shape[3]))
        
        result = downsample_save_chunk_mode(
            source_arr, dest_arr, out_slices,
            downsampling_factors, 'raw', True
        )
        
        assert result == 0
        # Verify gaussian filter was called
        mock_filter.assert_called_once()
        

@pytest.fixture
def multiscale_result(temp_zarr_array):
    """Create a multiscale zarr array and return the result for testing."""
    
    z_root, temp_dir = temp_zarr_array
    
    # Initialize real dask client
    dask_client = initialize_dask_client('local')
    
    try:
        # Mock wait to return immediately for testing
        with patch('src.to_multiscale.wait', return_value=([], [])):
            create_multiscale(z_root, dask_client, 'raw', False, True)
        
        yield z_root, dask_client, temp_dir
        
    finally:
        # Cleanup
        dask_client.close()
        

class TestMultiscaleMetadata:
    """Test the create_multiscale function."""
    
    def test_multiscale_metadata_structure(self, multiscale_result):
        """Test that create_multiscale produces correct metadata structure."""
        z_root, dask_client, temp_dir = multiscale_result
        
        # Test metadata structure
        assert 'multiscales' in z_root.attrs
        multiscales = z_root.attrs['multiscales']
        
        # Should be a list with one multiscale
        assert isinstance(multiscales, list)
        assert len(multiscales) == 1
        
        multiscale = multiscales[0]
        
        # Check required fields
        assert 'version' in multiscale
        assert 'axes' in multiscale
        assert 'datasets' in multiscale
        
        # Check axes structure
        axes = multiscale['axes']
        assert len(axes) == 4  # c, z, y, x
        expected_axes = ['c', 'z', 'y', 'x']
        for i, axis in enumerate(axes):
            assert 'name' in axis
            assert 'type' in axis
            assert axis['name'] == expected_axes[i]
        
        # Check datasets - should have s0 and at least s1
        datasets = multiscale['datasets']
        assert len(datasets) >= 2
        
        # Verify all dataset paths exist
        for dataset in datasets:
            path = dataset['path']
            assert path in z_root, f"Dataset {path} not found in zarr group"
        
        # Check that s1, s2, etc. were created
        assert 's1' in z_root
    
    def test_multiscale_array_shapes(self, multiscale_result):
        """Test that multiscale arrays have correct shapes."""
        z_root, dask_client, temp_dir = multiscale_result
        axes_order = [axis['name'].lower() for axis in z_root.attrs['multiscales'][0]['axes']]
        
        # Check that each level is downsampled by factor of 2
        level = 1
        spatial_shape = z_root[f's{level-1}'].shape[1:]

        while all([dim > 32 for dim in spatial_shape]):  # Check s1, s2, s3
            level_name = f's{level}'
            arrprev_shape = z_root[f's{level-1}'].shape

            scaling_factors = get_downsampling_factors(arrprev_shape, axes_order, high_aspect_ratio=True)
            if level_name in z_root:
                level_array = z_root[level_name]
                expected_shape = tuple(
                    dim // sc  # Don't downsample channel dimension
                    for dim, sc in zip(arrprev_shape,scaling_factors)
                )

                spatial_shape = level_array.shape[1:]
                level += 1
                assert level_array.shape == expected_shape, f"Level {level} has wrong shape"
    
    def test_multiscale_coordinate_transformations(self, multiscale_result):
        """Test coordinate transformations in multiscale metadata."""
        z_root, dask_client, temp_dir = multiscale_result
        
        multiscales = z_root.attrs['multiscales']
        datasets = multiscales[0]['datasets']
        axes_order = [axis['name'].lower() for axis in z_root.attrs['multiscales'][0]['axes']]
        
        # Check coordinate transformations for each level
        for i, dataset in enumerate(datasets):
            transforms = dataset['coordinateTransformations']
            assert len(transforms) == 2  # scale and translation
            
            # Check scale transformation
            scale_transform = transforms[0]
            assert scale_transform['type'] == 'scale'
            scale = scale_transform['scale']
            assert len(scale) == 4  # c, z, y, x
            
            # Channel scale should always be 1.0
            assert scale[0] == 1.0
            
            # Spatial scales should increase based on actual downsampling factors
            if i > 0:
                base_scale = datasets[0]['coordinateTransformations'][0]['scale']
                
                # Calculate cumulative scaling factors up to this level
                cumulative_factors = [1.0] * len(axes_order)  # Start with no scaling
                
                for level in range(1, i + 1):
                    # Get the shape of the previous level to calculate downsampling factors
                    prev_level_path = datasets[level - 1]['path']
                    if prev_level_path in z_root:
                        prev_shape = z_root[prev_level_path].shape
                        level_factors = get_downsampling_factors(prev_shape, axes_order, high_aspect_ratio=True)
                        
                        # Apply these factors to cumulative scaling
                        for dim_idx, factor in enumerate(level_factors):
                            cumulative_factors[dim_idx] *= factor
                
                # Check that the scale matches expected cumulative scaling
                for dim in [1, 2, 3]:  # z, y, x dimensions
                    expected_scale = base_scale[dim] * cumulative_factors[dim]
                    assert abs(scale[dim] - expected_scale) < 0.001, f"Level {i}, dim {dim}: expected {expected_scale}, got {scale[dim]}"
            
            # Check translation transformation
            translation_transform = transforms[1]
            assert translation_transform['type'] == 'translation'
            translation = translation_transform['translation']
            assert len(translation) == 4  # c, z, y, x
    
    def test_multiscale_data_values(self, multiscale_result):
        """Test that downsampled data has reasonable values."""
        z_root, dask_client, temp_dir = multiscale_result
        
        s0_data = z_root['s0'][:]
        
        # Check that s1 exists and has data
        if 's1' in z_root:
            s1_data = z_root['s1'][:]
            
            assert s1_data.min() >= 0  
            assert s1_data.max() <= s0_data.max() * 1.1 
            assert s1_data.mean() > 0 
            assert s1_data.shape[0] == s0_data.shape[0]  
            
            assert s1_data.std() > 0  
            
            mean_ratio = s1_data.mean() / s0_data.mean()
            assert 0.5 <= mean_ratio <= 2.0, f"Mean ratio {mean_ratio} is outside reasonable range"
    
    def test_multiscale_version_and_format(self, multiscale_result):
        """Test multiscale version and format compliance."""
        z_root, dask_client, temp_dir = multiscale_result
        
        multiscales = z_root.attrs['multiscales']
        multiscale = multiscales[0]
        
        # Check version
        assert 'version' in multiscale
        version = multiscale['version']
        assert version in ['0.4', '0.5']  # Should be a supported version
        
        # Check axes compliance
        axes = multiscale['axes']
        axis_names = [axis['name'] for axis in axes]
        assert axis_names == ['c', 'z', 'y', 'x']  # Expected order
        
        axis_types = [axis['type'] for axis in axes]
        expected_types = ['channel', 'space', 'space', 'space']
        assert axis_types == expected_types
        