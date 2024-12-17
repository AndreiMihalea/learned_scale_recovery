#!/usr/bin/env bash

# Get the hostname of the machine
hostname=$(hostname)

default_data_path="/mnt/datadisk/"
host_data_path=""

# Set the host data path based on the hostname
if [[ "$hostname" == "nemodrive1" ]]; then
    host_data_path="/mnt/datadisk/"
elif [[ "$hostname" == "nemodrive0" ]]; then
    host_data_path="/mnt/storage/workspace/"
elif [[ "$hostname" == "aimas-nvidia" ]]; then
    host_data_path="/raid/"
fi

echo $host_data_path