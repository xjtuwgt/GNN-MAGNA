import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from pandas import DataFrame
import pandas as pd
from time import time
from pathlib import Path
import torch
import torch.nn as nn
import shutil

def remove_all_files(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def save_to_json(data: DataFrame, file_name, orient='split'):
    start = time()
    data.to_json(file_name, orient=orient)
    print('Saving {} records in {:.2f} seconds'.format(data.shape[0], time() - start))

def load_json_as_data_frame(file_name: str, orient='split'):
    start = time()
    data = pd.read_json(file_name, orient=orient)
    print('Data loading takes {:.2f} seconds'.format(time() - start))
    return data

def save_to_HDF(data: DataFrame, file_name):
    start = time()
    data.to_hdf(path_or_buf=file_name,
                mode='w', key='df', format='table', data_columns=True)
    print('Saving {} records in {:.2f} seconds'.format(data.shape[0], time() - start))

def load_HDF_as_data_frame(file_name: str):
    start = time()
    data = pd.read_hdf(path_or_buf=file_name)
    print('Data loading takes {:.2f} seconds'.format(time() - start))
    return data

def save_model(model, step, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        model_state_dict[key] = model_state_dict[key].cpu()
    torch.save({
        'step': step,
        'model_state_dict': model_state_dict
    }, model_path)

def load_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    step = checkpoint['step']
    return model, step

def save_checkpoint_model(model, step, optimizer, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        model_state_dict[key] = model_state_dict[key].cpu()
    opt_state_dict = optimizer.state_dict()
    for key in opt_state_dict:
        opt_state_dict[key] = opt_state_dict[key].cpu()
    torch.save({
        'step': step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': opt_state_dict,
    }, model_path)

def load_checkpoint_model(model, optimizer, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['step']
    return model, optimizer, epoch
