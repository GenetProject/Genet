#!/bin/bash

python3 check_virtual_env.py
if [ $? -eq 0 ]; then
    echo "Please use a python virtual environment! Installation stopped."
    exit
fi

echo "Install python packages..."
pip install --upgrade pip --no-cache-dir
pip install tensorflow-gpu==1.15
pip install tensorboard==1.15.0
pip install tensorboard-plugin-wit==1.8.0
pip install tflearn==0.5.0
pip install numba==0.53.1
pip install gym==0.18.0
pip install stable-baselines[mpi]==2.10.1
pip install pandas==1.1.5
pip install tqdm==4.62.2
pip install matplotlib==3.3.4
pip install visdom==0.1.8.9
pip install bayesian-optimization==1.2.0

# dependencies for abr emulation
pip install selenium==3.141.0
pip install PyVirtualDisplay==2.0
# pip install xvfbwrapper
# pip install torch
