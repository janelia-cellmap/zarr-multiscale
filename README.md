to create multiscale, first install poetry project and python dependencies: 
1. cd PATH_TO_POETRY_PROJECT_DIRECTORY/
2. poetry install

Input format:
zarr group with ome-ngff metadata

array naming convention within group: ``sN`` - where N is the resolution level

run script using cli:

3. ``poetry run python3 src/to_multiscale.py  --workers=NUMBER_OF_DASK_WORKERS --data_origin=raw/segmentations --cluster=lsf/local --src=PATH_TO_ZARR_GROUP_WITH_OME-NGFF_METADATA``
