to create multiscale, first install poetry project and python dependencies: 
1. cd PATH_TO_POETRY_PROJECT_DIRECTORY/
2. poetry install


run script using cli:
3. poetry run python3 src/to_multiscale.py  --workers=NUMBER_OF_DASK_WORKERS --data_origin=labels/raw --cluster=lsf/local --src=PATH_TO_ZARR_ARRAY_WITH_NGFF_METADATA
