# GENET: Automatic Curriculum Generation for Learning Adaptation in Networking

## Installation

### Operating system information
Simulation and training experiments were done in a Ubuntu server with
- **Kernal version**: #184-Ubuntu SMP Thu Mar 24 17:48:36 UTC 2022
- **CPU INFO**:
    - Architecture:        x86_64
    - CPU(s):              32
    - Model name:          Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz


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

## Installation

```bash
cd Genet
bash install.sh
```

## Unseen synthetic environments (Figure 9)

### ABR
```bash
cd fig_reproduce/fig9
bash run.sh
# Please wait for 5 minutes to let the testing output finish writing.
# Please find genet_fig_upload/fig9/fig9_abr.png
```

### CC
```bash
cd Genet # cd into the project root
python src/simulator/evaluate_synthetic_traces.py --save-dir results/evaluate_synthetic_dataset
python src/plot_scripts/plot_syn_dataset.py
```

## Reproduce Figure 13

### ABR
```bash
cd fig_reproduce/fig13
bash run.sh
# Please wait for 15 minutes to let the testing output finish writing
# Please find genet_fig_upload/fig13/fig13_abr_fcc.png and 
# genet_fig_upload/fig13/fig13_abr_norway.png
```

### CC
```bash
cd Genet # cd into the project root
python src/simulator/evaluate_synthetic_traces.py --save-dir results/evaluate_synthetic_dataset
python src/plot_scripts/plot.py
```


## Learning curves (Figure 18)
### ABR
```bash
cd Genet
bash src/drivers/abr/train_udr3.sh
bash src/drivers/abr/train_genet.sh
bash src/drivers/abr/train_cl1.sh
# TODO: add plot scripts
```
### CC
```bash
cd Genet
bash src/drivers/cc/train_udr3.sh
bash src/drivers/cc/train_genet.sh
bash src/drivers/cc/train_cl1.sh
bash src/drivers/cc/train_cl2.sh
bash src/drivers/cc/train_cl3.sh
# TODO: add plot scripts
```

<!-- ## Traces -->
<!--  -->
<!-- ### Real Traces -->
<!--  -->
<!-- Real traces are recorded on Pantheon platform and they can be downloaded from -->
<!-- [Pantheon](https://pantheon.stanford.edu/measurements/node/). There are three -->
<!-- connection types: cellular, ethernet, and wifi. The path to store them is -->
<!-- `Genet/data/${connection_type}` -->
<!--  -->
<!-- ### Syntheic Traces -->
<!--  -->
<!-- Generated by `Genet/src/simulator/trace.py` -->
<!--  -->
<!-- ## Configuration files -->
<!--  -->
<!-- The configurations are stored at `Genet/config/train` -->


### Rule-based baselines

- BBR: [paper](https://www.cis.upenn.edu/~cis553/files/BBR.pdf),
  [code](https://github.com/google/bbr),
  [implentation in simulator](src/simulator/network_simulator/bbr.py)
- Copa:
  [paper](https://www.usenix.org/system/files/conference/nsdi18/nsdi18-arun.pdf),
  [code](https://github.com/venkatarun95/genericCC)
- Cubic:
  [paper](https://www.cs.princeton.edu/courses/archive/fall16/cos561/papers/Cubic08.pdf),
  [code](https://git.kernel.org/pub/scm/linux/kernel/git/netdev/net-next.git/tree/net/ipv4/tcp_cubic.c),
  [implentation in simulator](src/simulator/network_simulator/cubic.py)
- PCC-Vivace:
  [paper](https://www.usenix.org/system/files/conference/nsdi18/nsdi18-dong.pdf),
  [code](https://github.com/PCCproject/PCC-Uspace),
  [implentation in simulator](src/simulator/network_simulator/pcc/vivace/vivace_latency.py)
