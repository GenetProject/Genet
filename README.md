# GENET: Automatic Curriculum Generation for Learning Adaptation in Networking

## Installation

### Operating system information
Simulation and training experiments were done in a Ubuntu server with
- **Kernal version**: #184-Ubuntu SMP Thu Mar 24 17:48:36 UTC 2022
- **CPU INFO**:
    - Architecture:        x86_64
    - CPU(s):              32
    - Model name:          Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz
So the installation below only runs on Ubuntu.

### Download the source code

```bash
git clone git@github.com:GenetProject/Genet.git
```

### Set up python virtual environment
- Python3 Virtual environment is highly recommended. Select one of the
  following to set up a python3 virtual environment. 
  - [venv](https://docs.python.org/3.7/library/venv.html) only
  ```bash
  python3 -m venv genet
  echo "[absolute path]/Genet/src" > genet/lib/[python version]/site-packages/genet.pth
  source genet/bin/activate
  ```
  - [virtualenv](https://virtualenv.pypa.io/en/latest/) only
  ```bash
  virtualenv -p python3 genet
  echo "[absolute path]/Genet/src" > genet/lib/[python version]/site-packages/genet.pth
  source genet/bin/activate
  ```
  - [virtualenv](https://virtualenv.pypa.io/en/latest/) and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
  ```bash
  mkvirtualenv -a Genet/ -p python3 genet
  workon genet
  add2virtualenv src/
  ```
- Now the virtual environment is activated.

## Install dependency

```bash
cd Genet
bash install.sh
```

## Unseen synthetic environments (Figure 9)
We choose Figure 9 to reproduce because it is the first evaluation figure 
which shows how Genet training improves models' performance on unseen
environments. Example figures are [here](/fig_reproduce/fig9).


### ABR
```bash
cd fig_reproduce/fig9
bash run.sh
# Please wait for 5 minutes to let the testing output finish writing.
# Please find fig_reproduce/fig9/fig9_abr.png
```

### CC
```bash
cd Genet # cd into the project root

# time usage: about 60 min
python src/simulator/evaluate_synthetic_traces.py \
  --save-dir results/cc/evaluate_synthetic_dataset \
  --dataset-dir data/cc/synthetic_dataset
# Please find fig_reproduce/fig9/fig9_cc.png
python src/plot_scripts/plot_syn_dataset.py
```
### LB
```bash
cd genet-lib-fig-upload
python rl_test.py --saved_model="results/testing_model/udr_1/model_ep_49600.ckpt" # example output: [-4.80, 0.07]
python rl_test.py --saved_model="results/testing_model/udr_2/model_ep_44000.ckpt" # example output: [-3.87, 0.08]
python rl_test.py --saved_model="results/testing_model/udr_3/model_ep_25600.ckpt" # example output: [-3.57, 0.07]
python rl_test.py --saved_model="results/testing_model/adr/model_ep_20200.ckpt" # example output: [-3.02, 0.04]
# Please find fig_reproduce/fig9/fig9_lb.png
python analysis/fig9_lb.py
```

## Reproduce Figure 13
We choose Figure 13 to reproduce because it is the first evaluation figure 
which shows how Genet training improves models' generalizability. Example
figures are at [here](/fig_reproduce/fig13).

### ABR
```bash
cd Genet/fig_reproduce/fig13
bash run.sh
# Please wait for 15 minutes to let the testing output finish writing
# Please find fig_reproduce/fig13/fig13_abr_fcc.png and 
# fig_reproduce/fig13/fig13_abr_norway.png
```

### CC
```bash
cd Genet # cd into the project root
python src/plot_scripts/plot_ethernet_bars.py
python src/plot_scripts/plot_cellular_bars.py

# Please find fig_reproduce/fig13/fig13_cc_ethernet.png and 
# fig_reproduce/fig13/fig13_cc_cellular.png
```


## Learning curves (Figure 18)
We choose Figure 18 to reproduce because it shows how Genet helps model 
training. Example figures are [here](/fig_reproduce/fig18).


### CC
Please download models.tar.gz from [here](https://drive.google.com/file/d/1QxMLyffHlox8r6aSpVj37JEQ4_dyb2iN/view?usp=sharing)
and untar it under ```Genet/```
```bash
cd Genet

bash src/drivers/cc/run_for_learning_curve.sh
python src/plot_scripts/plot_learning_curve.py

# Train from scratch? run the following commands
bash src/drivers/cc/train_udr3.sh
bash src/drivers/cc/train_genet.sh
bash src/drivers/cc/train_cl1.sh
bash src/drivers/cc/train_cl2.sh
bash src/drivers/cc/train_cl3.sh
```

### ABR
```bash
cd Genet

# Train from scratch? run the following commands
bash src/drivers/abr/train_udr3.sh
bash src/drivers/abr/train_genet.sh
bash src/drivers/abr/train_cl1.sh
```

## Emulation (Figure 17)

## ABR
```bash
cd Genet/src/emulator/abr/video_server
python video_server.py

# in a new terminal
cd Genet
bash src/drivers/abr_emulation.sh
```

### CC
Please clone [this repository](https://github.com/zxxia/pantheon) and follow
the instructions in README to install pantheon.
```bash
cd Genet && cd ..
git clone https://github.com/zxxia/pantheon
cd pantheon/drivers/post_nsdi/
bash run_real_traces_ethernet.sh # run emulation over ethernet traces
bash run_real_traces_cellular.sh # run emulation over cellular traces
cd Genet
python src/plot_scripts/plot_scatter.py
```
