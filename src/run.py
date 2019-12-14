"""
Run batch_runner.py in parallel

Adapted from code by user225312
https://stackoverflow.com/questions/4256107/running-bash-commands-in-python
"""

import multiprocessing
from joblib import Parallel, delayed
import subprocess

num_cores = multiprocessing.cpu_count()

def run_command(i):
    command = "python batch_runner.py --config al.config_rl_ac --job_id {} --save_path ../log/data/".format(i)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error

if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(run_command)(i) for i in range(0,101))