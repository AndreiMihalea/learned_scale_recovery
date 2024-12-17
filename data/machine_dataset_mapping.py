"""
This module is used to establish the data paths in order to replace the file
names in dataloaders with the correct paths from the current machine.
"""

import socket


hostname = socket.gethostname()

# The default data corresponds to the path generated from the data 
# pre-processing scripts
default_data_path = '/mnt/datadisk/'
host_data_path = ''

if hostname == 'nemodrive1':
    host_data_path = '/mnt/datadisk/'
elif hostname == 'nemodrive0':
    host_data_path = '/mnt/storage/workspace/'
elif hostname == 'aimas-nvidia':
    host_data_path = '/raid/'