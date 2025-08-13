
run script using cli:

3. ``poetry run python3 src/to_multiscale.py  --workers=NUMBER_OF_DASK_WORKERS --data_origin=raw/segmentations --cluster=lsf/local --src=PATH_TO_ZARR_GROUP_WITH_OME-NGFF_METADATA``


## Installation

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and install dependencies
git clone https://github.com/janelia-cellmap/zarr-multiscale.git
cd zarr-multiscale
pixi install


## Input format
zarr group with ome-ngff metadata

array naming convention within group: ``sN`` - where N is the resolution level

## Show help documentation
pixi run help

# Run the script using cli
``pixi run python3 src/to_multiscale.py --workers=NUMBER_OF_DASK_WORKERS --data_origin=raw/segmentations --cluster=local --src=PATH_TO_ZARR_GROUP_WITH_OME-NGFF_METADATA``

For lsf cluster:

``bsub -n 1 -J multiscale 'pixi run python3 src/to_multiscale.py --workers=NUMBER_OF_DASK_WORKERS --data_origin=raw/segmentations --cluster=lsf --src=PATH_TO_ZARR_GROUP_WITH_OME-NGFF_METADATA --log-dir=dask_workers_logs'``
