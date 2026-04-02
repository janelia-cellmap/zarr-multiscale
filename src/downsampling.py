from typing import Tuple
from xarray_multiscale import windowed_mean, windowed_mode
#from xarray_multiscale.reducers import windowed_mode_countless, windowed_mode_scipy
import zarr
import scipy.ndimage as ndi

def upscale_slice(slc: slice, factor: int):
    """
    Returns input slice coordinates. 

    Args:
        slc : output slice
        factor : upsampling factor
    Returns:
        slice: source array slice
    """
    return slice(slc.start * factor, slc.stop * factor, slc.step)

def downsample_save_chunk_mode(
        source: zarr.Array, 
        dest: zarr.Array, 
        out_slices: Tuple[slice, ...],
        downsampling_factors: Tuple[int, ...],
        data_origin: str,
        antialiasing : bool):
    
    """
    Downsamples source array slice and writes into destination array. 

    Args:
        source : source zarr array that needs to be downsampled
        dest : destination zarr array that contains downsampled data
        out_slices : part of the destination array that would contain the output data
        downsampling_factors : tuple that contains downsampling factors. dim(downsampling_factors) must match the shape of the source array
        data_origin : affects which downsampling method is used. Accepts two values: segmentation or raw
    """
    
    in_slices = tuple(upscale_slice(out_slice, fact) for out_slice, fact in zip(out_slices, downsampling_factors))
    source_data = source[in_slices]
    # only downsample source_data if it is not all 0s
    if not (source_data == 0).all():
        if data_origin == 'segmentations':
            ds_data = windowed_mode(source_data, window_size=downsampling_factors)
        elif data_origin == 'raw':
            if antialiasing:
                # blur data in chunk before downsampling to reduce aliasing of the image 
                # conservative Gaussian blur coeff: 2/2.5 = 0.8
                sigma = [0 if factor == 1 else factor/2.5 for factor in downsampling_factors]
                filtered_data = ndi.gaussian_filter(source_data, sigma=sigma)
                ds_data = windowed_mean(filtered_data, window_size=downsampling_factors)
            else:
                ds_data = windowed_mean(source_data, window_size=downsampling_factors)

        dest[out_slices] = ds_data
    return 0

def get_downsampling_factors(shape, axes_order, min_ratio=0.5, max_ratio=2.0, high_aspect_ratio=False):
    """
    Calculate adaptive downsampling factors based on aspect ratios.
    
    Args:
        shape: Array shape (c, z, y, x)
        axes_order: List of axis names ['c', 'z', 'y', 'x']
        min_ratio: Minimum allowed ratio (default 0.5)
        max_ratio: Maximum allowed ratio (default 2.0)
    
    Returns:
        List of downsampling factors
    """
    if high_aspect_ratio==False:
        return [ 1 if axis in ['c', 't'] else 2 for axis in axes_order]
    else:
        # Get spatial dimensions (skip channel and time)
        spatial_dims = list((axis, shape[i]) for i, axis in enumerate(axes_order) if axis not in ['c', 't'])
        
        axes, dimensions = zip(*spatial_dims)
    
        if len(dimensions) == 1:
            factors = [2]
        else:
            ratios = []
            for i, dim in enumerate(dimensions):
                # Calculate ratios of current dimension to all others
                # for example for 3D:  [(z/y, z/x),(y/z, y/x),(x/z, x/y)]
                dim_ratios = tuple(dim / dimensions[j] for j in range(len(dimensions)) if j != i)
                ratios.append(dim_ratios)
            
            # Determine downsampling factors for each spatial dimension
            factors = []
            for (i,dim_ratios) in enumerate(ratios):
                # Check if both ratios are within acceptable range
                if all(ratio >= max_ratio for ratio in dim_ratios):
                    factors = [1,]*len(ratios)
                    factors[i] = 2
                    break
                elif all(ratio <= min_ratio for ratio in dim_ratios):
                    factors = [2,]*len(ratios)
                    factors[i] = 1
                    break
                else:
                    factors.append(2)
        
        spatial_factors = {k : v for k,v in zip(axes, factors)}
        return tuple(1 if axis in ['c', 't'] else spatial_factors[axis] for axis in axes_order)