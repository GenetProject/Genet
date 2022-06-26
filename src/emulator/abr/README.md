# ABR emulator
We provide two options:
1. Run the full emulation.
2. Replot our emulation results. 

Since the full emulation running takes more than a day, the second option is faster for replotting.

# Run the full emulation
## Install
```bash
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get -y install mahimahi xvfb chromium-chromedriver python3-pip python3-tk
pip3 install virtualenv

virtualenv tf_venv
source tf_venv/bin/activate

pip3 install numpy tensorflow==1.15.0 selenium pyvirtualdisplay numba torch tflearn xvfbwrapper
```

## Go to folder and download the chromedriver for linux:
```bash
cd Genet/src/abr/emulator/pensieve/virtual_browser/abr_browser_dir

chrome_version=$(google-chrome --version | awk '{print $3}')
wget https://chromedriver.storage.googleapis.com/${chrome_version}/chromedriver_linux64.zip
unzip chromedriver_linux64.zip (cover the old one if needed)
```

## Run a ABR emulation example

Open one terminal for video server
```bash
cd Genet/src/abr/emulator/pensieve/video_server
python video_server.py  --port=8000
```

Open another terminal for virtual browser
```bash
cd Genet/src/abr/emulator
bash pensieve/drivers/run_mahimahi_emulation_ADR.sh  --port=8000 > /dev/null 2>&1 &
```

# Replot our emulation results
## Fig.17 (c) data
```bash
cd analysis
python print_each_dim_fcc.py

# Output of bitrate, rebuf: ['sim_BBA: bitrate: 1.2% rebuf: 0.05848', 
#                            'sim_RobustMPC: bitrate: 1.22% rebuf: 0.03195', 
#                            'sim_udr_1: bitrate: 1.2% rebuf: 0.03384', 
#                            'sim_udr_2: bitrate: 1.04% rebuf: 0.01955', 
#                            'sim_udr_3: bitrate: 1.1% rebuf: 0.02367', 
#                            'sim_adr: bitrate: 1.11% rebuf: 0.01486']
```
Note: we will remove the Fugu point in the camera ready version since we only
have its results on FCC trace, not on Norway.

## Fig.17 (d) data
```bash
cd analysis
python print_each_dim_norway.py

# Output of bitrate, rebuf: ['sim_BBA: bitrate: 1.03% rebuf: 0.07658',
#                            'sim_RobustMPC: bitrate: 1.05% rebuf: 0.05053', 
#                            'sim_udr_1: bitrate: 1.04% rebuf: 0.07323', 
#                            'sim_udr_2: bitrate: 0.96% rebuf: 0.04276', 
#                            'sim_udr_3: bitrate: 0.95% rebuf: 0.04796', 
#                            'sim_adr: bitrate: 0.95% rebuf: 0.04498']
```

## Put the above output to plot
```bash
cd analysis
python mahi_results_two_dim.py
```
