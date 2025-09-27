"""
This module is used to establish the data paths in order to replace the file
names in dataloaders with the correct paths from the current machine.
"""

import socket


hostname = socket.gethostname()

# The default data corresponds to the path generated from the data 
# pre-processing scripts
default_data_path = '/mnt/datadisk/andreim/kitti'
host_data_path = ''

if hostname == 'nemodrive1':
    host_data_path = '/mnt/datadisk/andreim/kitti'
elif hostname in ['nemodrive0', 'aimas']:
    host_data_path = '/mnt/storage/workspace/andreim/kitti'
elif hostname == 'aimas-nvidia':
    host_data_path = '/raid/andreim/kitti'
elif hostname == 'andrei-pc':
    host_data_path = '/HDD_2TB/storage/KITTI'
