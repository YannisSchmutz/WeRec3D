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