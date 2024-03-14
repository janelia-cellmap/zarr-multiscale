import json
import os
from xarray_multiscale import multiscale, windowed_mean,windowed_mode
import dask.array as da
import zarr
import cluster_wrapper as cw
import time
import copy 
import numpy as np
from dask.array.core import slices_from_chunks
from dask.distributed import Client
from numcodecs import Zstd

src_store = zarr.NestedDirectoryStore('')
src_root = zarr.open_group(store=src_store, mode = 'a')

####################################
def create_multiscale(z_root, chunks, client, num_workers, comp):
    # store original array in a new .zarr file as an arr_name scale
    z_attrs = copy.deepcopy(dict(z_root.attrs)) #add_multiscale_metadata(z_root)
    #optimal_chunksize =  optimal_dask_chunksize(z_root['s0'], 30000)
    # sn_data = da.from_array(src_root[arr_name], chunks = optimal_chunksize)
    # dataset = dest_root.create_dataset(arr_name, shape=src_root[arr_name].shape, chunks=src_root[arr_name].chunks, dtype=src_root[arr_name].dtype)
    # da.store(sn_data, dataset, lock = False)
    print(list(z_root.attrs))
    print(list(z_root.array_keys(recurse = True)))
    i = 0
    s0 = list(z_root.attrs['multiscales'][0]['datasets'][i]['coordinateTransformations'][0]['scale'])
    tr0 = list(z_root.attrs['multiscales'][0]['datasets'][i]['coordinateTransformations'][1]['translation'])
    b = 1
    
    z_arr_shape = z_root["s0"].shape
    while all(x > 64 for x in z_arr_shape):#(z_arr_shape > tuple(i*b for i in (64,64,64))):
        print(i)
        if i != 0:

            opt_chunksize = optimal_dask_chunksize(z_root[f's{i-1}'], chunks, 2000)
            print(f"main array, opt chunk_size: {opt_chunksize}")
            sn_data = da.from_array(z_root[f's{i-1}'], chunks = opt_chunksize)
            res_level = multiscale(sn_data, windowed_mode, 2, chunks = opt_chunksize)[1]

            start_time = time.time()
            dataset = z_root.create_dataset(f's{i}', shape=res_level.shape, chunks=chunks, dtype=res_level.dtype, compressor=comp, dimension_separator='/')


            slab_size = (106, int(sn_data.shape[1] /16), -1)
            slices_src = slices_from_chunks(sn_data.rechunk(slab_size).chunks)
                        
            slab_reduced_size = (int(slab_size[0] / 2), int(slab_size[1] / 2), -1)
            slices_target = slices_from_chunks(res_level.data.rechunk(slab_reduced_size).chunks)   
            print(len(list(zip(slices_src, slices_target))))
            
            for batch in batch_list(list(zip(slices_src, slices_target)), 10):
                
                client.cluster.scale(num_workers)

                futures = []
                for (sl_src, sl_target) in batch:

                    darr_in = sn_data[sl_src]
                    opt_chunksize_slab = optimal_dask_chunksize(darr_in, chunks, 2000, 2)
                                    
                    res_level_slab = multiscale(darr_in, windowed_mode, 2, chunks = opt_chunksize_slab)[1]
                    futures.append(da.store(res_level_slab.data, dataset, regions=sl_target, lock=False, return_stored=False, compute=False))
                
                client.compute(futures, sync=True)
            
                client.cluster.scale(0)
            
            # client.cluster.scale(num_workers)
            # da.store(res_level.data, dataset, lock=False)
            # client.cluster.scale(0)

            
            print(f"computation time, s_{i}: {time.time() - start_time}")
            z_arr_shape = res_level.shape


            sn = [dim * pow(2, i) for dim in s0]
            trn = [(dim * (pow(2, i - 1) -0.5))+tr for (dim, tr) in zip(s0, tr0)]

            print(trn)
            z_attrs['multiscales'][0]['datasets'].append({'coordinateTransformations': [{"type": "scale",
                        "scale": sn}, {"type" : "translation", "translation" : trn}],
                        'path': f's{i}'})
        i += 1
    
    z_root.attrs['multiscales'] = z_attrs['multiscales']

# def optimal_dask_chunksize(arr, target_chunks, max_dask_chunk_num):
#     #calculate number of chunks within a zarr array.
#     chunk_dims = target_chunks
#     chunk_num= np.prod(arr.shape)/np.prod(chunk_dims) 
    
#     # 1. Scale up chunk size (chunksize approx = 1GB)
#     scaling = 1
#     while np.prod(chunk_dims)*arr.itemsize*pow(scaling, 3)/pow(10, 6) < 700 :
#         scaling += 1

#     # 3. Number of chunks should be < 50000
#     while (chunk_num / pow(scaling,3)) > max_dask_chunk_num:
#         scaling +=1

#     # 2. Make sure that chunk dims < array dims
#     while any([ch_dim > 3*arr_dim/4 for ch_dim, arr_dim in zip(tuple(dim * scaling for dim in chunk_dims), arr.shape)]):#np.prod(chunks)*arr.itemsize*pow(scaling,3) > arr.nbytes:
#         scaling -=1

#     if scaling == 0:
#         scaling = 1
#     return tuple(dim * scaling for dim in chunk_dims) 

def optimal_dask_chunksize(arr, target_chunks, max_dask_chunk_num, scale_dim=3):
    #calculate number of chunks within a zarr array.
    chunk_dims = target_chunks
    chunk_num= np.prod(arr.shape)/np.prod(chunk_dims) 
    
    # 1. Scale up chunk size (chunksize approx = 1GB)
    scaling = 1
    while np.prod(chunk_dims)*arr.itemsize*pow(scaling, 3)/pow(10, 6) < 700 :
        scaling += 1
    
    print(scaling)

    # 2. Number of chunks should be < 50000
    while (chunk_num / pow(scaling,3)) > max_dask_chunk_num:
        scaling +=1
        
    print(scaling)

    # 3. Make sure that chunk dims < array dims
    while any([ch_dim > 3*arr_dim/4 for ch_dim, arr_dim in zip(tuple(dim * scaling for dim in chunk_dims[-scale_dim:]), arr.shape[-scale_dim:])]):#np.prod(chunks)*arr.itemsize*pow(scaling,3) > arr.nbytes:
        scaling -=1
        
    print(scaling)

    if scaling == 0:
        scaling = 1
            
    #anisotropic scaling
    scaling_dims = np.ones(len(chunk_dims), dtype=int)
    
    scaling_dims[-scale_dim:] = scaling
    print(scaling_dims)
        
    return tuple(dim * scale_dim for dim, scale_dim  in zip(chunk_dims, scaling_dims)) 
    

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

def batch_list(arr, n=1):
    l = len(arr)
    for ndx in range(0, l, n):
        yield arr[ndx:min(ndx + n, l)]

if __name__ == '__main__':
    # store_multiscale = cw.cluster_compute("local")(create_multiscale)
    # store_multiscale(src_root,(64, 64, 64),  Zstd(level=6))
    
    client = Client(cw.get_cluster("lsf", 60))
    text_file = open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w")
    text_file.write(str(client.dashboard_link))
    text_file.close()
    create_multiscale(src_root,(106, 106, 106), client, 11, Zstd(level=6))
 
