# ABR emulator
We provide two options:
1. Run the full emulation.
2. Replot our emulation results. 

Since the full emulation running takes more than a day, the second option is faster for replotting.

# Run the full emulation
## Install
```
sudo apt update
sudo apt install mahimahi
sudo apt-get install xvfb
sudo apt install chromium-chromedriver


sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install python3-pip
sudo pip3 install virtualenv
sudo apt-get install python3-tk

virtualenv tf_venv
source tf_venv/bin/activate

pip3 install numpy tensorflow==1.15.0 selenium pyvirtualdisplay numba torch tflearn xvfbwrapper
```

## Check google-chrome version:
```
google-chrome --version

ex: xx.x.xxx.xxx
```

## Find the matched chromedriver:
```
https://chromedriver.storage.googleapis.com/index.html?path=xx.x.xxx.xxx/
```

## Go to folder and download the chromedriver for linux:
```
cd pensieve/virtual_browser/abr_browser_dir
wget [chromedriver_linux64.zip link]
unzip chromedriver_linux64.zip (cover the old one if needed)
```

## Run a ABR emulation example

Open one terminal for video server
```
cd pensieve/video_server
python video_server.py  --port=8000
```

Open another terminal for virtual browser
```
bash pensieve/drivers/run_mahimahi_emulation_ADR.sh  --port=8000 > /dev/null 2>&1 &
```
# Replot our emulation results
## Fig.17 (c) data
```
cd analysis
python print_each_dim_fcc.py

# Output of bitrate, rebuf: ['sim_BBA: bitrate: 1.2% rebuf: 0.05848', 'sim_RobustMPC: bitrate: 1.22% rebuf: 0.03195', 'sim_udr_1: bitrate:
 1.2% rebuf: 0.03384', 'sim_udr_2: bitrate: 1.04% rebuf: 0.01955', 'sim_udr_3: bitrate: 1.1% rebuf: 0.02367', 'sim_adr: bitrate: 1.11% rebuf: 0.01486']
```
Note: we will remove the Fugu point in the camera ready version since we only have its results on FCC trace, not on Norway.

## Fig.17 (d) data
```
cd analysis
python print_each_dim_norway.py

# Output of bitrate, rebuf: ['sim_BBA: bitrate: 1.03% rebuf: 0.07658', 'sim_RobustMPC: bitrate: 1.05% rebuf: 0.05053', 'sim_udr_1: bitrate:
 1.04% rebuf: 0.07323', 'sim_udr_2: bitrate: 0.96% rebuf: 0.04276', 'sim_udr_3: bitrate: 0.95% rebuf: 0.04796', 'sim_adr: bitrate: 0.95% rebuf: 0.04498']

```

## Put the above output to plot
```
cd analysis
python mahi_results_two_dim.py
```