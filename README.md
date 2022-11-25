# GENET: Automatic Curriculum Generation for Learning Adaptation in Networking

## Installation

### Operating system information
Ubuntu 18.04. A large VM is preferred, e.g., reproducing Figure 9 CC takes
about 20 minutes on a VM with 96 vCPUs or 1 hour on a VM with 32 vCPUs. We
assume a VM with 32 vCPUs, 64G memory and 32G SSD storage is used for the
instructions below.

### Python Version
The repository is only tested under python3.6.9.

### Download the source code

```bash
git clone https://github.com/GenetProject/Genet.git 
```

### Download models
Please download ```models.tar.gz``` to ```Genet/```.
```bash
cd Genet
wget https://www.dropbox.com/s/39x6im0di4c8dos/models.tar.gz?dl=0
tar -xf models.tar.gz
```

### Install apt packages

```bash
cd Genet
bash install.sh
```

### Set up python virtual environment
Python3 Virtual environment is highly recommended.
[venv](https://docs.python.org/3.7/library/venv.html) only
```bash
cd Genet && cd ..  # make sure the virtual env is at the same level of Genet/
python3 -m venv genet
echo "$(pwd)/Genet/src" > genet/lib/python3.6/site-packages/genet.pth
source genet/bin/activate
  ```
  <!-- - [virtualenv](https://virtualenv.pypa.io/en/latest/) only -->
  <!-- ```bash -->
  <!-- virtualenv -p python3 genet -->
  <!-- echo "[absolute path]/Genet/src" > genet/lib/[python version]/site-packages/genet.pth -->
  <!-- source genet/bin/activate -->
  <!-- ``` -->
  <!-- - [virtualenv](https://virtualenv.pypa.io/en/latest/) and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) -->
  <!-- ```bash -->
  <!-- mkvirtualenv -a Genet/ -p python3 genet -->
  <!-- workon genet -->
  <!-- add2virtualenv src/ -->
  <!-- ``` -->
- Now the virtual environment is activated.

### Install python dependency
Time usage: 2~3min on a VM with 32 vCPUs
```bash
cd Genet
# activate virtual env
bash install_python_dependency.sh
```

## Unseen synthetic environments (Figure 9)
We choose Figure 9 to reproduce because it is the first evaluation figure 
which shows how Genet training improves models' performance on unseen
environments. Example figures are [here](/fig_reproduce/fig9).


### ABR
Time usage: 1~2 min on a VM with 32 vCPUs.
Please find ```fig_reproduce/fig9/fig9_abr.png```
```bash
cd Genet/fig_reproduce/fig9
bash run.sh
```

### CC
Please find ```fig_reproduce/fig9/fig9_cc.png```. The result difference may be 
caused by randomness.

To get a fast and slightly different but with the correct trend,
the following command run models with 1 seed on 50 synthetic traces, 

Time usage: 2 min on a VM with 32 vCPUs.
```bash
cd Genet # cd into the project root
python src/simulator/evaluate_synthetic_traces.py \
  --save-dir results/cc/evaluate_synthetic_dataset \
  --dataset-dir data/cc/synthetic_dataset \
  --fast
python src/plot_scripts/plot_syn_dataset.py
```

To get a complete results,  the following commands run models with 5 different
seeds on ~500 synthetic traces

Time usage: 60 min on a VM with 32 vCPUs.
```bash
cd Genet # cd into the project root
python src/simulator/evaluate_synthetic_traces.py \
  --save-dir results/cc/evaluate_synthetic_dataset \
  --dataset-dir data/cc/synthetic_dataset
python src/plot_scripts/plot_syn_dataset.py
```

### LB
Time usage: ~10 min on a VM with 32 vCPUs.

Please find ```fig_reproduce/fig9/fig9_lb.png```
```bash
cd Genet/genet-lb-fig-upload

# example output: [-4.80, 0.07]
python rl_test.py --saved_model="results/testing_model/udr_1/model_ep_49600.ckpt"

# example output: [-3.87, 0.08]
python rl_test.py --saved_model="results/testing_model/udr_2/model_ep_44000.ckpt"

# example output: [-3.57, 0.07]
python rl_test.py --saved_model="results/testing_model/udr_3/model_ep_25600.ckpt"

# example output: [-3.02, 0.04]
python rl_test.py --saved_model="results/testing_model/adr/model_ep_20200.ckpt"

python analysis/fig9_lb.py
```

## Generalizability(Figure 13)
We choose Figure 13 to reproduce because it is the first evaluation figure 
which shows how Genet training improves models' generalizability. Example
figures are at [here](/fig_reproduce/fig13).

### ABR
Time usage: ~5min on a VM with 32 vCPUs.

Please find ```fig_reproduce/fig13/fig13_abr_fcc.png``` and 
```fig_reproduce/fig13/fig13_abr_norway.png```.

```bash
cd Genet/fig_reproduce/fig13
bash run.sh
```

### CC
Time usage: ~5min on a VM with 32 vCPUs.

Please find ```fig_reproduce/fig13/fig13_cc_ethernet.png``` and 
```fig_reproduce/fig13/fig13_cc_cellular.png```.

```bash
cd Genet # cd into the project root
python src/plot_scripts/plot_bars_ethernet.py
python src/plot_scripts/plot_bars_cellular.py
```


## Emulation (Figure 17)

## ABR
Please follow the [README](https://github.com/GenetProject/Genet/tree/main/src/emulator/abr#readme) under ```src/emulator/abr/```

### CC
Please run the following commands in to install pantheon.
```bash
deactivate # only deactivate when "genet" python3 virtual environment is activated
cd Genet && cd ..  # under the same parenet folder of Genet/
git clone https://github.com/zxxia/pantheon
cd pantheon
git fetch && git checkout artifact  # swtich to the correct branch
./tools/fetch_submodules.sh  # fetch cc algorithms as submodules
./tools/install_deps.sh  # install apt dependencies
```
Install python dependencies.
```bash
cd pantheon && cd ..
virtualenv -p python2 pantheon_venv  # create py2 venv for pantheon
echo "$(pwd)/pantheon" > pantheon_venv/lib/python2.7/site-packages/pantheon.pth
source pantheon_venv/bin/activate  # activate py2 venv
cd pantheon
./tools/install_py2_deps.sh  # install apt dependencies
src/experiments/setup.py --install-deps --schemes "cubic bbr copa vivace aurora vivace_loss vivace_latency"
src/experiments/setup.py --setup --schemes "cubic bbr copa vivace aurora vivace_loss vivace_latency"
```

Download ```data.tar.gz```to ```pantheon/```.
```bash 
cd pantheon 
wget https://www.dropbox.com/s/q5v8s9v8himb5gf/data.tar.gz?dl=0
tar -xf data.tar.gz
```.

Run the following commands in to emulate.
Expected time usage: 19hr
```bash
cd pantheon
# run emulation over ethernet traces
bash drivers/post_nsdi/run_real_traces_ethernet_rule_based.sh 
bash drivers/post_nsdi/run_real_traces_ethernet_genet.sh 
bash drivers/post_nsdi/run_real_traces_ethernet_rl.sh
bash drivers/post_nsdi/run_real_traces_ethernet_cl.sh
```
<!-- bash run_real_traces_cellular.sh # run emulation over cellular traces -->
<!-- cd Genet -->
<!-- python src/plot_scripts/plot_scatter.py -->

## Learning curves (Figure 18)
Figure 18 is optional because the ramp-up or convergence speed on a
learning curve is not one of our primary claims, i.e., Genet leads to better
asymptotic performance and generalization.
Example figures are [here](/fig_reproduce/fig18). Training from
scratch is optional.


### CC
Running pretrained model.

Expected time usage: 5hr on a VM with 32 vCPUs by sequentially running the
following scrits. Please find ```Genet/fig_reproduce/fig18_cc_example.png```.

```bash
source genet/bin/activate
cd Genet

bash src/drivers/cc/run_for_learning_curve.sh
python src/plot_scripts/plot_learning_curve.py
```

Training model from scratch is optinal
Expected time usage: 21hr on a VM with 32 vCPUs by sequentially running the
following scripts.
```bash
bash src/drivers/cc/train_udr3.sh
bash src/drivers/cc/train_genet.sh
bash src/drivers/cc/train_cl1.sh
bash src/drivers/cc/train_cl2.sh
bash src/drivers/cc/train_cl3.sh
```

<!-- ### ABR -->
<!-- ```bash -->
<!-- cd Genet -->
<!--  -->
<!-- # Train from scratch? run the following commands -->
<!-- bash src/drivers/abr/train_udr3.sh -->
<!-- bash src/drivers/abr/train_genet.sh -->
<!-- bash src/drivers/abr/train_cl1.sh -->
<!-- ``` -->

## FAQ
1. CUDA driver error

    If the following cuda driver error message shows up, please ignore for now.
    The final results are not affected by the error message.
    ```bash
     E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: UNKNOWN ERROR
    (genet) ubuntu@reproduce-genet:~/Genet/genet-lb-fig-upload$ python rl_test.py --saved_model="results/testing_model/udr_1/model_ep_49600.ckpt"
    2022-06-23 20:46:00.130224: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: UNKNOWN ERROR
    ```
