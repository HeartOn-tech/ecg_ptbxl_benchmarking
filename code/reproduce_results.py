import os
import argparse
from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *


def main(record_base_path):
    datafolder_ptbxl = os.path.join(record_base_path, 'ptbxl')  # '../data/ptbxl/'
    datafolder_icbeb = os.path.join(record_base_path, 'ICBEB')  # '../data/ICBEB/'
    outputfolder = '../output/'
    mode = 'eval'
                 # 'results' - только таблица результатов
                 # 'estim' - только оценка эффективности моделей в зависимости от порога и таблица результатов
                 # 'eval' - подготовка данных (prepare()) и расчет оценок классификации по моделям,
                 # 'predict' - только выполнение предсказания по обученным моделям,
                 # 'fit' - выполнение обучения моделей,
                 # 'finetune' - дообучение моделей (только fastai-модели)

    models = [
        conf_fastai_xresnet1d101,
        conf_fastai_resnet1d_wang,
        conf_fastai_lstm,
        conf_fastai_lstm_bidir,
        conf_fastai_fcn_wang,
        conf_fastai_inception1d,
        conf_wavelet_standard_nn
        ]

    # Evaluation module additional options
    eval_params = {
        'data_types': ['train', 'valid', 'test'], # Data types
        'save_mlb_file' : False,           # save mlb.pkl file in prepare()
        'N_thrs' : 2,                      # number of threshold criterion
        'use_train_valid_for_thr' : False, # True: use train and valid data for threshold evaluation, False: use train data only for threshold evaluation
        'add_train_folds' : True,          # add train folds for analysis - True/False
        'save_pdf_files' : False,          # save pdf files with graphs - True/False
        'save_raw_txt' : False             # save output csv files with raw data (by labels) - True/False
        }

    if eval_params['use_train_valid_for_thr']: # base: train & valid, estim: test
        file_types = ['folds_t_v']
    else:                                      # base: train, estim: valid, test
        file_types = ['folds_t']

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################
    use_PTBXL = True

    if use_PTBXL:
        data_name = 'ptbxl'
        experiments = [
            ('exp0', 'all'),
            ('exp1', 'diagnostic'),
            ('exp1.1', 'subdiagnostic'),
            ('exp1.1.1', 'superdiagnostic'),
            ('exp2', 'form'),
            ('exp3', 'rhythm')
           ]
        exps = []

        for exp_name, task in experiments:
            exps.append(exp_name)

            if mode != 'results':
                e = SCP_Experiment(data_name,
                                   exp_name,
                                   task,
                                   datafolder_ptbxl,
                                   outputfolder,
                                   models,
                                   mode = mode,
                                   eval_params = eval_params)

                if mode != 'estim':
                    e.prepare()
                if mode != 'eval' and mode != 'estim':
                    e.perform()

                e.evaluate()

        #generate greate summary table
        #if mode != 'estim':
        utils.generate_summary_table(data_name,
                                     exps,
                                     outputfolder,
                                     None,
                                     file_types)

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################
    use_ICBEB = True

    if use_ICBEB:
        data_name = 'ICBEB'
        exp_name = 'exp_ICBEB'
        task = 'all'

        if mode != 'results':
            e = SCP_Experiment(data_name,
                               exp_name,
                               task,
                               datafolder_icbeb,
                               outputfolder,
                               models,
                               mode = mode,
                               eval_params = eval_params)

            if mode != 'estim':
                e.prepare()
            if mode != 'eval' and mode != 'estim':
                e.perform()

            e.evaluate()

        # generate greate summary table
        #if mode != 'estim':
        utils.generate_summary_table(data_name,
                                     [exp_name],
                                     outputfolder,
                                     None,
                                     file_types)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='path saving test record file')
    args = parser.parse_args()
    record_base_path = args.recording_path

    main(record_base_path)
