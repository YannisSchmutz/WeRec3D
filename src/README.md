# WeRec3D


## Install requirements and venv
This description is targeted for MacOS or Linux operating systems and uses Python3.11.

Make sure piptools is installed
```
pip3.11 install pip-tools
pip3.11 install ipykernel
```
Create requirements file
```
cd requirements/
make requirements.txt
```

Create venv
```
# Go to src/
python3.11 -m venv venv_werec3d
```
Activate venv
```
source venv_werec3d/bin/activate
```
Update
```
pip install --upgrade pip
```
To deactivate later:
```
deactivate
```
Execute outside of venv:
```
python3.11 -m ipykernel install --user --name=venv_werec3d
```

Install requirements to venv:
Pip-sync will install/update/uninstall everything to match the things defined in requirements.txt
(Execute outside of venv)
```
# Locally
pip-sync --python-executable venv_werec3d/bin/python requirements/requirements.txt
```


## Procedure for reproducibility
1. Obtain and prepare ERA5 Data from the very scratch. As the necessary files created in this step are added to the git repo, one can skip it and proceed with step 2.
```
cd data_source/
python download_create_daymean_sets.py -k <uid:key>
python reduce_size_granularity.py
# Download ETOPO1_Ice_g_geotiff.tif, save it to
# https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/georeferenced_tiff/
python create_elevation_data.py
python create_land_sea_mask.py
```
2. Create climatology and anomaly data.
```
mkdir temporal_adjusted
python adjust_temporal.py
```







## Handy Docker Commands

Start docker container
```
sudo docker-compose up --build -d
```
Access docker container
```
sudo docker exec -it werec3d bash
```

Stop docker container
```
sudo docker stop werec3d
```