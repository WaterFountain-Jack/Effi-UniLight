# Effi-UniLight

# docker
We recommend using dockers to build the cityflow platform.
```python
docker pull zyr17/unlight
sudo docker run -it --name EffiUni-Light zyr17/unilight /bin/bash 
```

# build 
Download and copy the zip file into the created container, then unzip the zip file after entering the container.
```python
docker cp EffiUni-Light.zip EffiUni-Light:/
docker exec -it EffiUni-Light bash
unzip Effi-UniLight-main.zip
cd Effi-UniLight-main
unzip data.zip
pip install -r requirements.txt
```

# run & test
You can decide which dataset to run by changing the parameters.
The model is tested once after each training session.
```python
#'jinan' 'hangzhou' 'newyork16_3 'newyork28_7' 'manhattan' 'SH1' 'SH2'
python main.py --data 'jinan'
```

# Details

## Environment
The definition of environment is in `envs/env_Heterogeneous.py`.

## Dataset
You can find the datasets in `/data`.

## Agents
You can find the agent in `/agent/Effi_UniLight` and the detailed structural design of the model in `/agent/Effi_UniLight_core.py`.

## Arguments and Configs
We parse the parameters via `config/config.py`.

## Summary
We count the results of the experiment via `summary.py`.
