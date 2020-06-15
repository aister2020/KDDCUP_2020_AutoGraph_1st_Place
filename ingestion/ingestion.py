# pylint: disable=logging-fstring-interpolation, broad-except
"""ingestion program for autoWSL"""
import os
from os.path import join
import sys
from sys import path
import argparse
import time
import pandas as pd
import yaml
from filelock import FileLock

from common import get_logger, init_usermodel

import timing
from timing import Timer
from dataset import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Verbosity level of logging:
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def _here(*args):
    """Helper function for getting the current directory of this script."""
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, *args))


def write_start_file(output_dir):
    """write start file"""
    start_filepath = os.path.join(output_dir, 'start.txt')
    lockfile = os.path.join(output_dir, 'start.txt.lock')
    ingestion_pid = os.getpid()

    with FileLock(lockfile):
        with open(start_filepath, 'w') as ftmp:
            ftmp.write(str(ingestion_pid))

    LOGGER.info('===== Finished writing "start.txt" file.')


class IngestionError(RuntimeError):
    """Model api error"""


def _parse_args():
    root_dir = _here(os.pardir)
    #mac or linux
    #default_dataset_dir = join(root_dir, "sample_data/a/train.data")
    #windows
    default_dataset_dir = join(root_dir, "sample_data\\e\\train.data")
    default_output_dir = join(root_dir, "sample_result_submission")
    default_ingestion_program_dir = join(root_dir, "ingestion_program")
    default_code_dir = join(root_dir, "code_submission")
    default_score_dir = join(root_dir, "scoring_output")
    default_temp_dir = join(root_dir, 'temp_output')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset (containing "
                             "e.g. adult.data/)")
    parser.add_argument('--output_dir', type=str,
                        default=default_output_dir,
                        help="Directory storing the predictions. It will "
                             "contain e.g. [start.txt, predictions, end.yaml]"
                             "when ingestion terminates.")
    parser.add_argument('--ingestion_program_dir', type=str,
                        default=default_ingestion_program_dir,
                        help="Directory storing the ingestion program "
                             "`ingestion.py` and other necessary packages.")
    parser.add_argument('--code_dir', type=str,
                        default=default_code_dir,
                        help="Directory storing the submission code "
                             "`model.py` and other necessary packages.")
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help="Directory storing the scoring output "
                             "e.g. `scores.txt` and `detailed_results.html`.")
    parser.add_argument('--temp_dir', type=str,
                        default=default_temp_dir,
                        help="Directory storing the temporary output."
                             "e.g. save the participants` model after "
                             "trainning.")

    args = parser.parse_args()
    LOGGER.debug(f'Parsed args are: {args}')
    LOGGER.debug("-" * 50)
    if (args.dataset_dir.endswith('run/input') and
            args.code_dir.endswith('run/program')):
        LOGGER.debug("Since dataset_dir ends with 'run/input' and code_dir "
                     "ends with 'run/program', suppose running on "
                     "CodaLab platform. Modify dataset_dir to 'run/input_data'"
                     " and code_dir to 'run/submission'. "
                     "Directory parsing should be more flexible in the code of"
                     " compute worker: we need explicit directories for "
                     "dataset_dir and code_dir.")

        args.dataset_dir = args.dataset_dir.replace(
            'run/input', 'run/input_data')
        args.code_dir = args.code_dir.replace(
            'run/program', 'run/submission')

        # Show directories for debugging
        LOGGER.debug(f"sys.argv = {sys.argv}")
        LOGGER.debug(f"Using dataset_dir: {args.dataset_dir}")
        LOGGER.debug(f"Using output_dir: {args.output_dir}")
        LOGGER.debug(
            f"Using ingestion_program_dir: {args.ingestion_program_dir}")
        LOGGER.debug(f"Using code_dir: {args.code_dir}")
    return args


def _init_python_path(args):
    path.append(args.ingestion_program_dir)
    path.append(args.code_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)


def _train_predict(umodel, dataset, timer, n_class, schema):
    # Train the model
    data = dataset.get_data()

    with timer.time_limit('train_predict'):
        predictions = umodel.train_predict(
            data, timer.get_all_remain()['train_predict'], n_class, schema)

    return predictions


def _finalize(args, timer):
    # Finishing ingestion program
    end_time = time.time()

    time_stats = timer.get_all_stats()
    for pname, stats in time_stats.items():
        for stat_name, val in stats.items():
            LOGGER.info(f'the {stat_name} of duration in {pname}: {val} sec')

    overall_time_spent = timer.get_overall_duration()

    # Write overall_time_spent to a end.yaml file
    end_filename = 'end.yaml'
    content = {
        'ingestion_duration': overall_time_spent,
        'time_stats': time_stats,
        'end_time': end_time}

    with open(join(args.output_dir, end_filename), 'w') as ftmp:
        yaml.dump(content, ftmp)
        LOGGER.info(
            f'Wrote the file {end_filename} marking the end of ingestion.')

        LOGGER.info("[+] Done. Ingestion program successfully terminated.")
        LOGGER.info(f"[+] Overall time spent {overall_time_spent:5.2} sec")

    # Copy all files in output_dir to score_dir
    # os.system(
    #     f"cp -R {os.path.join(args.output_dir, '*')} {args.score_dir}")
    LOGGER.debug(
        "Copied all ingestion output to scoring output directory.")

    LOGGER.info("[Ingestion terminated]")


def _write_predict(output_dir, prediction):
    """prediction should be list"""
    os.makedirs(output_dir, exist_ok=True)
    prediction = pd.Series(prediction, name='label')
    LOGGER.debug(f'prediction shape: {prediction.shape}')
    prediction.to_csv(
        join(output_dir, 'predictions'), index=False, header=True)


def _init_timer(time_budgets):
    timer = Timer()
    timer.add_process('train_predict', time_budgets, timing.RESET)
    LOGGER.debug(
        f"init time budget of train_predict: {time_budgets} "
        f"mode: {timing.RESET}")
    return timer

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    fontsize = int(15*7/len(classes))
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='red', fontsize=fontsize, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.show()


def main():
    """main entry"""
    LOGGER.info('===== Start ingestion program.')
    # Parse directories from input arguments
    LOGGER.info('===== Initialize args.')
    args = _parse_args()
    _init_python_path(args)
    
    accuarcies = []
    overall_time_spents = []
    valid_accuarcies1 = []
    valid_accuarcies2 = []
    accuarcies1 = []
    accuarcies2 = []
    
#    for file in ['a','b','c','d','e']:
#    for file in ['a']:
    for file in ['a','b','d','e']:
        LOGGER.info(f'===== Start Dataset {file}')
        root_dir = _here(os.pardir)
        args.dataset_dir = join(root_dir, "data/"+file+"/train.data")
    
        # write_start_file(args.output_dir)
    
        LOGGER.info('===== Load data.')
        dataset = Dataset(args.dataset_dir)
        time_budget = dataset.get_metadata().get("time_budget")*10
#        time_budget = 100
        n_class = dataset.get_metadata().get("n_class")
        schema = dataset.get_metadata().get("schema")
        LOGGER.info(f"Time budget: {time_budget}")
    
        LOGGER.info("===== import user model")
        umodel = init_usermodel()
    
        LOGGER.info("===== Begin training user model")
        timer = _init_timer(time_budget)
        predictions,valid_acc1,valid_acc2,preds1,preds2 = _train_predict(umodel, dataset, timer, n_class, schema)
        valid_accuarcies1.append(valid_acc1)
        valid_accuarcies2.append(valid_acc2)
        LOGGER.info(f"valid accuracy1:{valid_acc1}")
        LOGGER.info(f"valid accuracy2:{valid_acc2}")
        
        
        accuarcy1 = (dataset.test_label['label'].values==preds1).sum()/predictions.shape[0]
        accuarcy2 = (dataset.test_label['label'].values==preds2).sum()/predictions.shape[0]
        accuarcy = (dataset.test_label['label'].values==predictions).sum()/predictions.shape[0]
        
        accuarcies1.append(accuarcy1)
        LOGGER.info(f"test accuracy1:{accuarcy1}")
        accuarcies2.append(accuarcy2)
        LOGGER.info(f"test accuracy2:{accuarcy2}")
        accuarcies.append(accuarcy)
        LOGGER.info(f"test accuracy:{accuarcy}")
       # _write_predict(args.output_dir, predictions)
    
        # _finalize(args, timer)
        
        overall_time_spent = timer.get_overall_duration()
        LOGGER.info(f"time spent:{overall_time_spent}")
        overall_time_spents.append(overall_time_spent)
        
    
    LOGGER.info(f"valid accuarcies1:{valid_accuarcies1}")
    LOGGER.info(f"valid accuarcies2:{valid_accuarcies2}")
    LOGGER.info(f"test accuarcies1:{accuarcies1}")
    LOGGER.info(f"test accuarcies2:{accuarcies2}")
    LOGGER.info(f"test accuarcies:{accuarcies}")
    LOGGER.info(f"time spents:{overall_time_spents}")
if __name__ == "__main__":
    main()
