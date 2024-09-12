import os
from typing import Tuple
from xarray_multiscale import multiscale, windowed_mean, windowed_mode
from xarray_multiscale.reducers import windowed_mode_countless, windowed_mode_scipy
import dask.array as da
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

@nb.njit
def index_to_coords(idx: int, shape: Tuple[int]):
    ndim = len(shape)
    result = np.zeros(ndim, dtype="int")
    init = idx
    strides = make_strides(shape)
    for region_idx in range(0, ndim - 1):
        result[region_idx] = init // strides[region_idx]
        init -= result[region_idx] * strides[region_idx]
    result[-1] = init
    return result


@nb.njit
def make_strides(shape: Tuple[int]):
    ndim = len(shape)
    result = np.ones(ndim, dtype="int")
    for d in range(ndim - 2, -1, -1):
        result[d] = shape[d + 1] * result[d + 1]
    return result


@nb.njit
def create_stencil(array_shape: Tuple[int], region_shape: Tuple[int]):
    ndim = len(array_shape)

    result_shape = 1
    for x in region_shape:
        result_shape *= x

    array_strides = make_strides(array_shape)

    result = np.zeros(result_shape, dtype="int64")

    for lidx in range(0, result_shape):
        shift = 0
        region_idx = index_to_coords(lidx, region_shape)
        for c in range(ndim):
            shift += region_idx[c] * array_strides[c]
        result[lidx] = shift
    return result


@nb.jit
def reduce(arr, region_shape: Tuple[int], reduction, num_reductions):
    array_shape = np.array(arr.shape)
    region_shape = np.array(region_shape)
    region_size = np.prod(region_shape)
    flat = arr.ravel()
    stencil = create_stencil(array_shape, region_shape)

    partitions = np.zeros(num_reductions, dtype="int")
    partitions[0] = len(flat) // region_size

    for n in range(1, num_reductions):
        partitions[n] = partitions[n - 1] // region_size

    output = np.zeros(partitions.sum(), dtype=arr.dtype)

    region_grid_shape = array_shape // region_shape
    array_strides = make_strides(array_shape)
    region_grid_strides = array_strides * region_shape

    for idx in range(partitions[0]):
        region_coord = index_to_coords(idx, region_grid_shape)
        stencil_shift = (region_coord * region_grid_strides).sum()
        output[idx] = reduction(flat[stencil + stencil_shift])

    return output

@nb.jit()
def mode_nb(v):
    sorted = np.sort(v)
    local_frequency = 1
    candidate_frequency = 1
    local_mode = sorted[0]
    candidate_mode = sorted[0]

    for index in range(1, len(v)):
        if sorted[index] == local_mode:
            local_frequency += 1
        else:
            local_frequency = 1
            local_mode = sorted[index]
        
        if local_frequency >= candidate_frequency:
            candidate_frequency = local_frequency
            candidate_mode = local_mode
        if candidate_frequency >= len(v) // 2:
            return candidate_mode
    return candidate_mode

def downsample_save_chunk_mode(
        source: zarr.Array, 
        dest: zarr.Array, 
        out_slices: Tuple[slice, ...],
        downsampling_factors: Tuple[int, ...],
        data_origin: str):
    
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
            ds_data = windowed_mean(source_data, window_size=downsampling_factors)
        
        dest[out_slices] = ds_data
    return 1

def create_multiscale(z_root: zarr.Group, client: Client, num_workers: int, data_origin: str):
    
    """
    Creates multiscale pyramid and writes corresponding metadata into .zattrs 

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
    
    client.cluster.scale(num_workers)
    
    level = 1
    source_shape = z_root[f's{level-1}'].shape
    
    # continue downsampling if output array dimensions > 32 
    while all([dim > 32 for dim in source_shape]):
        print(source_shape)
        print(f'{level=}')
        source_arr = z_root[f's{level-1}']
        source_darr = da.from_array(source_arr, chunks=(-1,-1,-1))
        
        if data_origin == 'segmentations':
            res_level_template = multiscale(source_darr, windowed_mode, 2)[1].data
        elif data_origin == 'raw':
            res_level_template = multiscale(source_darr, windowed_mean, 2)[1].data

        # initialize output array
        dest_arr = z_root.require_dataset(
            f's{level}', 
            shape=res_level_template.shape, 
            chunks=source_arr.chunks, 
            dtype=res_level_template.dtype, 
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
            fut = client.map(lambda v: downsample_save_chunk_mode(source_arr, dest_arr, v, (2,2,2), data_origin), part)
            print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
            
            # wait for all the futures to complete
            result = wait(fut)
            print(f'Completed {len(part)} tasks in {time.time() - start}s')
            
        # calculate scale and transalation for n-th scale 
        sn = [dim * pow(2, level) for dim in base_scale]
        trn = [(dim * (pow(2, level - 1) -0.5))+tr for (dim, tr) in zip(base_scale, base_trans)]

        # store scale, translation
        z_attrs['multiscales'][0]['datasets'].append({'coordinateTransformations': [{"type": "scale",
                    "scale": sn}, {"type" : "translation", "translation" : trn}],
                    'path': f's{level}'})
        
        level += 1
        source_shape = res_level_template.shape
    
    # write multiscale metadata into .zattrs
    z_root.attrs['multiscales'] = z_attrs['multiscales']
        
@click.command()
@click.option('--src','-s',type=click.Path(exists = True),help='Input .zarr file location.')
@click.option('--workers','-w',default=100,type=click.INT,help = "Number of dask workers")
@click.option('--data_origin','-do',type=click.STRING,help='Different data requires different type of interpolation. Raw fibsem data - use \'raw\', for segmentations - use \'segmentations\'')
@click.option('--cluster', '-c', default='' ,type=click.STRING, help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'")
def cli(src, workers, data_origin, cluster):
    
    src_store = zarr.NestedDirectoryStore(src)
    src_root = zarr.open_group(store=src_store, mode = 'a')
    
    if cluster == '':
        print('Did not specify which instance of the dask client to use!')
        sys.exit(0)
    elif cluster == 'lsf':
        num_cores = 1
        cluster = LSFCluster(
            cores=num_cores,
            processes=num_cores,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="48:00",
            local_directory = "/scratch/$USER/"
            )
    
    elif cluster == 'local':
            cluster = LocalCluster()
    
    client = Client(cluster)
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)

    create_multiscale(z_root=src_root, client=client, num_workers=workers, data_origin=data_origin)
    
if __name__ == '__main__':
    cli()

