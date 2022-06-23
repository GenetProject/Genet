#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Update apt package lists..."
    sudo apt-get update

    echo "Install dependencies for stable-baselines2..."
    sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev

    echo "Install mahimahi for emulation..."
    sudo apt-get -y install mahimahi
    sudo sysctl -w net.ipv4.ip_forward=1

    echo "Install python3 venv for emulation..."
    sudo apt-get -y install python3-venv

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    echo "Mac OSX, not supported!"
    exit
else
    # Unknown.
    echo "Not supported!"
    exit
fi
