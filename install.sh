python3 check_virtual_env.py
if [ $? -eq 0 ]; then
    echo "Please use a python virtual environment! Installation stopped."
    exit
fi

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Update apt package lists..."
    sudo apt-get update

    echo "Install dependencies for stable-baselines2..."
    sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    echo "Mac OSX, not supported!"
    exit
else
    # Unknown.
    echo "Not supported!"
    exit
fi

echo "Install python packages..."
pip install tensorflow-gpu==1.15
pip install stable-baselines[mpi]
pip install pandas==1.1.5
pip install tqdm==4.62.2
pip install matplotlib==3.3.4
pip install visdom==0.1.8.9
