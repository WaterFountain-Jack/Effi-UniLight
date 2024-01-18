# Effi-UniLight

# docker
We use CityFlow as a traffic simulator. The version is the same as in `https://github.com/zyr17/UniLight`.
`git clone git@github.com:zyr17/UniLight.git`

# build 
We recommend using the mirror: `zyr17/unlight` to build the cityflow environment. Alternatively, you can build an environment yourself. Our code runs on Python 3.6.5 and will work on higher versions of Python, but compatibility is not guaranteed.
`pip install -r requirements.txt`

# run
You can decide which dataset to run by changing the parameters.
```python
#'jinan' 'hangzhou' 'newyork16_3 'newyork28_7' 'manhattan' 'SH1' 'SH2'
python main.py --data 'jinan'
```

# Details

## Environment
You can decide which dataset to run by changing the parameters

The definition of environment is in `envs/env_Heterogeneous.py`.

## Dataset
You can find the datasets in `/data`.

## Agents
You can find the agent in `/agent` and the detailed structural design of the model in `/agent/Effi_UniLight_core.py`.

## Arguments and Configs
We parse the parameters via `config/config.py`.

## Summary
We count the results of the experiment via `summary.py`.
