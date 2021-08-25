import torch
from pathlib import Path
import argparse
import os
import json

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

def save_model(model, model_save_path, step):
    if isinstance(model_save_path, Path):
        model_path = str(model_save_path)
    else:
        model_path = model_save_path
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        model_state_dict[key] = model_state_dict[key].cpu()
    torch.save({
        'step': step,
        'model_state_dict': model_state_dict
    }, model_path)

def load_model(model, model_load_path):
    if isinstance(model_load_path, Path):
        model_path = str(model_load_path)
    else:
        model_path = model_load_path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    step = checkpoint['step']
    return model, step

def save_config(args, save_path):
    argparse_dict = vars(args)
    with open(os.path.join(save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

def load_config(load_path):
    parser = argparse.ArgumentParser()
    with open(os.path.join(load_path, 'config.json'), 'rt') as fjson:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(fjson))
        args = parser.parse_args(namespace=t_args)
    return args

def remove_models(dir_name, best_model_name, extension='.pt'):
    all_files = os.listdir(dir_name)
    count = 0
    for file in all_files:
        if file.endswith(extension) and file != best_model_name:
            os.remove(os.path.join(dir_name, file))
            count = count + 1
    print('{} files end with {} have been removed'.format(count, extension))

if __name__ == '__main__':
    dir_name = '/Users/xjtuwgt/PycharmProjects/GraphTransformerNodeClassifcation/models/citeseerlr_0.0005lyer_2hs_8ho_3hi_256tk_5pd_-1ind_0.5att_0.6ed_0.1alpha_0.2decay_0.1'
    # files = os.listdir(dir_name)
    # print(files)
    remove_models(dir_name, best_model_name='124_vacc_0.646_tacc_0.646.pt')