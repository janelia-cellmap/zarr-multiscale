import os
from typing import Tuple
from xarray_multiscale import windowed_mean, windowed_mode
#from xarray_multiscale.reducers import windowed_mode_countless, windowed_mode_scipy
import zarr
import time
from numcodecs.abc import Codec
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client, wait, LocalCluster
from numcodecs import Zstd
from toolz import partition_all
from dask_jobqueue import LSFCluster
import click
import sys
import math
import scipy.ndimage as ndi
from dask_utils import initialize_dask_client


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

def create_multiscale(z_root: zarr.Group, client: Client, data_origin: str, antialiasing : bool):
    
    """
    Creates multiscale pyramid and write corresponding metadata into .zattrs 

    Args:
        z_root : parent group for source zarr array
        client : Dask client instance
        num_workers : Number of dask workers
        data_origin : affects which downsampling method is used. Accepts two values: 'segmentation' or 'raw'
    """
    # store original array in a new .zarr file as an arr_name scale
    z_attrs = z_root.attrs.asdict() 
    base_scale = z_attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
    base_trans = z_attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][1]['translation']
        
    level = 1
    source_shape = z_root[f's{level-1}'].shape
    axes_order = [axis['name'].lower() for axis in z_attrs['multiscales'][0]['axes']]
    scaling_factors = [ 1 if axis in ['c', 't'] else 2 for axis in axes_order]
    spatial_shape = [source_shape[dim] for dim, scaling in enumerate(scaling_factors) if scaling == 2]
    #continue downsampling if output array dimensions > 32 
    while all([dim > 32 for dim in spatial_shape]):
        print(f'{level=}')
        source_arr = z_root[f's{level-1}']
        
        dest_shape = [math.floor(dim / scaling) for dim, scaling in zip(source_arr.shape, scaling_factors)]

        # initialize output array
        dest_arr = z_root.require_dataset(
            f's{level}', 
            shape=dest_shape, 
            chunks=source_arr.chunks, 
            dtype=source_arr.dtype, 
            compressor=source_arr.compressor, 
            dimension_separator='/',
            fill_value=0,
            exact=True)
                
        assert dest_arr.chunks == source_arr.chunks
        out_slices = slices_from_chunks(normalize_chunks(source_arr.chunks, shape=dest_arr.shape))
        
        #break the slices up into batches, to make things easier for the dask scheduler
        out_slices_partitioned = tuple(partition_all(100000, out_slices))
        for idx, part in enumerate(out_slices_partitioned):
            print(f'{idx + 1} / {len(out_slices_partitioned)}')
            start = time.time()
            fut = client.map(lambda v: downsample_save_chunk_mode(source_arr, dest_arr, v, scaling_factors, data_origin, antialiasing), part)
            print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
            
            # wait for all the futures to complete
            result = wait(fut)
            print(f'Completed {len(part)} tasks in {time.time() - start}s')
            
        # calculate scale and transalation for n-th scale 
        sn = [dim * pow(2, level) if scaling==2 else dim for dim, scaling in zip(base_scale, scaling_factors)]
        trn = [round((dim * (pow(2, level - 1) -0.5))+tr, 3) if scaling==2 else tr
               for (dim, tr, scaling) in zip(base_scale, base_trans, scaling_factors)]

        # store scale, translation
        z_attrs['multiscales'][0]['datasets'].append({'coordinateTransformations': [{"type": "scale",
                    "scale": sn}, {"type" : "translation", "translation" : trn}],
                    'path': f's{level}'})
        
        level += 1
        spatial_shape = [dest_shape[dim] for dim, scaling in enumerate(scaling_factors) if scaling == 2]
    
    #write multiscale metadata into .zattrs
    z_root.attrs['multiscales'] = z_attrs['multiscales']
        
@click.command()
@click.option('--src','-s',type=click.Path(exists = True),help='Input .zarr file location.')
@click.option('--workers','-w',default=100,type=click.INT,help = "Number of dask workers")
@click.option('--data_origin','-do',type=click.STRING,help='Different data requires different type of interpolation. Raw fibsem data - use \'raw\', for segmentations - use \'segmentations\'')
@click.option('--cluster', '-c', default=None ,type=click.STRING, help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'")
@click.option('--log_dir', default = None, type=click.STRING,
    help="The path of the parent directory for all LSF worker logs.  Omit if you want worker logs to be emailed to you.")
@click.option('--antialiasing', '-aa', default=False, type=click.BOOL, help='Reduce aliasing of the image by blurring it with Gaussian filter before downsampling. Default: False')
def cli(src, workers, data_origin, cluster, log_dir, antialiasing):
    
    src_store = zarr.NestedDirectoryStore(src)
    src_root = zarr.open_group(store=src_store, mode = 'a')
    
    client = initialize_dask_client(cluster_type=cluster, log_dir=log_dir)

    client.cluster.scale(workers)
    create_multiscale(z_root=src_root, client=client, data_origin=data_origin, antialiasing=antialiasing)
    
if __name__ == '__main__':
    cli()

