#!/bin/bash

# Step 1: get the IP.
IP=$(hostname -I | awk '{print $1}')

# Step 2: Replace the IP in launch.json with the new IP.
sed -i "s/\"host\": \"[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*\"/\"host\": \"$IP\"/" .vscode/launch.json

echo "IP address updated in launch.json"