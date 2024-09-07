"""
####################################################################################
#####                         File name: utils.py                              #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####                   Common Util Function for Application                   #####
####################################################################################
"""
import os
import subprocess
from time import perf_counter

import torch


def timer(func):
    """Decorator that measures the execution time of a method."""
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print(f"[INFO]: Time taken to execute {func.__name__}: {end_time - start_time:.5f} seconds.")
        return result
    return wrapper


def get_gpu_memory_stats():
    """
    Function to Get GPU Memory Statistics
    """
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    print(result.stdout)

def get_device():
    """_summary_

    Returns:
        device: Returns torch.deice as "cuda" is cuda is available, else cpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"[INFO]: Using {device=}")
    return device

def get_total_gpu_memory():
    """_summary_

        Returns:
            gpu_memory_gb: Returns the total GPU memory in GB
    """
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes) /(2**30)
    print(f"[INFO]: Total available GPU memory: {gpu_memory_gb} (GB)")
    return gpu_memory_gb



def get_root_dir():
    """
    Get the absolute path of the root directory, which contains the 'requirements.txt' file.

    Returns:
        str: The absolute path to the root directory.
    """
    # Start at the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # Traverse upwards until we find the root directory containing 'requirements.txt'
    while True:
        # Check if 'requirements.txt' exists in the current directory
        if 'requirements.txt' in os.listdir(current_dir):
            return current_dir

        # Move one directory level up
        new_dir = os.path.dirname(current_dir)
        
        # If we've reached the root directory of the file system, stop
        if new_dir == current_dir:
            raise FileNotFoundError("Could not find 'requirements.txt' in any parent directory.")
        
        current_dir = new_dir
