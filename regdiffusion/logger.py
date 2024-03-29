import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Union

class LightLogger:
    ''' 
    A lightweight logger that runs completely in local
    
    This logger takes inspirations from w&b but runs completely in local 
    environment. Also, it supports logging multiple separated runs in 
    a single experiment. 
    
    Args:
        result_dir (str): Path to the dir to save all the logging files
        log_date (str): Within result_dir, logs from each date will be saved in 
        each subdirectory. This log_date variable provides a way to customize
        this setting.
    '''
    def __init__(self, result_dir: str = 'result_logs', 
                 log_date: str = None):
        if log_date is None:
            log_date = str(datetime.date.today())
        self.result_dir = result_dir
        self.log_dir = f'{result_dir}/{log_date}'
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.configs = {}
        self.mem = {}
        self.current_log = None
        self.logging_vars = set()
        self.early_stopping_min = None
    
    def set_configs(self, configs: Dict):
        """
        Save experiment configurations (a python dictionary) to memory for 
        future exportation

        Args:
            configs (dict): A python dictionary saving all the experimental 
                details. For example, the hyper parameters. 
        """
        self.configs = configs
    
    def start(self, note: str = None):
        """
        Start the logging of a new run within an experiment

        Args:
            note (str): A name for a new log stream. 
        """
        if note is None:
            note = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") 
            note += str(np.random.choice(100000))
        self.current_log = note
        self.mem[note] = {'time': note}
        for k in self.configs.keys():
            self.mem[note][k] = self.configs[k]
        self.mem[note]['current_step'] = -1
        self.mem[note]['log'] = {}
        return note
        
    
    def log(self, log_dict: Dict, step: int = None):
        """
        Log `log_dict` (a dictionary containing performance) at each step

        Args:
            log_dict (dict): A python dictionary (with performances) to log
            step (int): Which step to log on. 
        """
        if step is None:
            step = self.mem[self.current_log]['current_step'] + 1
        self.mem[self.current_log]['current_step'] = step
        self.mem[self.current_log]['log'][step] = {}
        for k in log_dict.keys():
            self.mem[self.current_log]['log'][step][k] = log_dict[k]
            self.logging_vars.add(k)

    def check_early_stopping(self, item: str, k: int = 10):
        end_idx = self.mem[self.current_log]['current_step']
        window = []
        start_idx = max(end_idx-k, 0)
        for idx in range(start_idx, end_idx):
            window.append(self.mem[self.current_log]['log'][idx][item])
        if len(window) < k:
            if len(window) != 0:
                self.early_stopping_min = min(window)
            return False
        else:
            if min(window) > self.early_stopping_min:
                return True
            else:
                self.early_stopping_min = min(window)
                return False
            
    def finish(self, save_now: bool = True):
        """
        End the logging of a run and save to a local file if `save_now` is 
        True.

        Args:
            save_now (bool): whether to dump the current log stream's memory
                to a local memory file. 
        """
        if save_now:
            with open(f'{self.log_dir}/{self.current_log}.json', 'w') as f:
                json.dump(self.mem[self.current_log], f)
        self.current_log = None
    
    def to_df(self, tidy: bool = True):
        """
        Convert saved logs to a pandas dataframe

        Args:
            tidy (bool): Whether to convert the df to a tidy format. Default
                is true.
        """
        export_df = pd.DataFrame(self.mem).transpose().reset_index()
        if tidy:
            export_df['steps'] = export_df['log'].map(lambda x: list(x.keys()))
            for v in self.logging_vars:
                export_df[v] = export_df['log'].map(
                    lambda x: [x[k].get(v) for k in x.keys()]
                )
            del export_df['log']
            export_df = export_df.explode(
                ['steps'] + list(self.logging_vars), ignore_index=True)
        return export_df
    
    def save(self, path: str):
        """
        Save all the logs to path

        Args:
            path (str): The file path used to save the logs. 
        """
        export = {}
        export['result_dir'] = self.result_dir
        export['log_dir'] = self.log_dir
        export['configs'] = self.configs
        export['mem'] = self.mem
        export['current_log'] = self.current_log
        export['logging_vars'] = list(self.logging_vars)
        with open(path, 'w') as f:
            json.dump(export, f)

def load_logger(path):
    ''' Load a saved log file to a LightLogger object
    
    Parameters
    ----------
    path: str
        path to the json file generated by LightLogger.save.
    '''
    with open(path, 'r') as f:
        logger_import = json.load(f)
    log_date = logger_import['log_dir'].replace(logger_import['result_dir']+'/', '')
    logger = LightLogger(logger_import['result_dir'], log_date=log_date)
    logger.set_configs(logger_import['configs'])
    logger.mem = logger_import['mem']
    logger.current_log = logger_import['current_log']
    logger.logging_vars = set(logger_import['logging_vars'])
    print("Loading logger complete!")
    return logger

