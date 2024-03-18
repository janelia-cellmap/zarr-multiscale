import json
import os
from typing import Literal, Tuple, Union
from distributed import wait
from xarray_multiscale import multiscale, windowed_mean, windowed_mode
from xarray_multiscale.reducers import windowed_mode_countless, windowed_mode_scipy
import dask.array as da
import zarr
import cluster_wrapper as cw
import time
from numcodecs.abc import Codec
import numpy as np
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client
from numcodecs import Zstd
from toolz import partition_all
import numba as nb
src_store = zarr.NestedDirectoryStore('/nrs/cellmap/zubovy/liver_zon_1_predictions/mito_membrane_postprocessed.zarr/inference')
src_root = zarr.open_group(store=src_store, mode = 'a')

def upscale_slice(slc: slice, factor: int):
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
        downsampling_factors: Tuple[int, ...]):
    
    in_slices = tuple(upscale_slice(out_slice, fact) for out_slice, fact in zip(out_slices, downsampling_factors))
    source_data = source[in_slices]
    # only downsample source_data if it is not all 0s
    if not (source_data == 0).all():
        # ds_data = windowed_mode(source_data, window_size=downsampling_factors)
        ds_data = reduce(source_data, (2,2,2), mode_nb, 1).reshape(tuple(s // 2 for s in source_data.shape))
        dest[out_slices] = ds_data
    return 1

def create_multiscale(z_root: zarr.Group, out_chunks: Tuple[int, ...], client: Client, num_workers: int, comp: Codec):
    # store original array in a new .zarr file as an arr_name scale
    z_attrs = z_root.attrs.asdict() 
    base_scale = z_attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
    base_trans = z_attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][1]['translation']
    num_levels = 8
    client.cluster.scale(num_workers)
    for level in range(1, num_levels - 1):
        print(f'{level=}')
        source_arr = z_root[f's{level-1}']
        source_darr = da.from_array(source_arr, chunks=(-1,-1,-1))
        res_level_template = multiscale(source_darr, windowed_mode, 2)[1].data

        start_time = time.time()
        dest_arr = z_root.require_dataset(
            f's{level}', 
            shape=res_level_template.shape, 
            chunks=out_chunks, 
            dtype=res_level_template.dtype, 
            compressor=comp, 
            dimension_separator='/',
            fill_value=0,
            exact=True)
        
        assert dest_arr.chunks == out_chunks
        out_slices = slices_from_chunks(normalize_chunks(out_chunks, shape=dest_arr.shape))
        # break the slices up into batches, to make things easier for the dask scheduler
        out_slices_partitioned = tuple(partition_all(100000, out_slices))
        for idx, part in enumerate(out_slices_partitioned):
            print(f'{idx + 1} / {len(out_slices_partitioned)}')
            start = time.time()
            fut = client.map(lambda v: downsample_save_chunk_mode(source_arr, dest_arr, v, (2,2,2)), part)
            print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
            # wait for all the futures to complete
            result = wait(fut)
            print(f'Completed {len(part)} tasks in {time.time() - start}s')
        sn = [dim * pow(2, level) for dim in base_scale]
        trn = [(dim * (pow(2, level - 1) -0.5))+tr for (dim, tr) in zip(base_scale, base_trans)]

        z_attrs['multiscales'][0]['datasets'].append({'coordinateTransformations': [{"type": "scale",
                    "scale": sn}, {"type" : "translation", "translation" : trn}],
                    'path': f's{level}'})
    
    z_root.attrs['multiscales'] = z_attrs['multiscales']
    

def add_multiscale_metadata(dest_root):
    #populate .zattrs
    path = os.path.join(os.path.abspath(os.sep), os.getcwd(), 'src/zarr_attrs_template.json' )
    print(path)
    f_zattrs_template = open(path )
    z_attrs = json.load(f_zattrs_template)
    
    #z_attrs['multiscales'][0]['axes'] = [{"name": axis, 
    #                                    "type": "space",
    #                                    "unit": "nanometer"} for axis in ['x', 'y', 'z']]
                                                                                
    z_attrs['multiscales'][0]['version'] = '0.4'
    z_attrs['multiscales'][0]['name'] = dest_root.name
    return z_attrs

if __name__ == '__main__':
    # store_multiscale = cw.cluster_compute("local")(create_multiscale)
    # store_multiscale(src_root,(64, 64, 64),  Zstd(level=6))
    from dask_jobqueue import LSFCluster
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
    client = Client(cluster)
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)

    out_chunks = (212,) * 3
    create_multiscale(z_root=src_root, out_chunks=out_chunks, client=client, num_workers=100, comp=Zstd(level=6))
 
