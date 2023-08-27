#!/bin/bash

# Step 1: Load the modules
module load cuda/11.8 
module load python/3.9.6
pip3 install -r requirements.txt

# Step 2: get the IP.
IP=$(hostname -I | awk '{print $1}')

# Step 2: Replace the IP in launch.json with the new IP.
sed -i "s/\"host\": \"[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*\"/\"host\": \"$IP\"/" .vscode/launch.json

echo "IP address updated in launch.json"