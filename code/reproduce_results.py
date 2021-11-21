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
    mode = 'estim'
                 # 'results' - только таблица результатов
                 # 'estim' - только оценка эффективности моделей в зависимости от порога и таблица результатов
                 # 'eval' - только расчет оценок классификации по моделям,
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

    # Data set
    # use_train_valid_for_thr: True: use train and valid data for threshold evaluation, False: use train data only for threshold evaluation
    use_train_valid_for_thr = False
    # save output csv files with raw data (by labels) - True/False
    save_eval_raw_txt = False
    data_types = ['train', 'valid', 'test']

    if use_train_valid_for_thr: # base: train & valid, estim: test
        file_types = ['train_valid_thr', 'test_t_v']
        file_types_suffix = 't_v_thr_t'
    else:                       # base: train, estim: valid, test
        file_types = ['train_thr', 'valid_t', 'test_t']
        file_types_suffix = 't_thr_v_t'

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
                                   use_train_valid_for_thr = use_train_valid_for_thr,
                                   save_eval_raw_txt = save_eval_raw_txt)

                if mode != 'eval' and mode != 'estim':
                    e.prepare()
                    e.perform()

                e.evaluate(data_types = data_types)

        #generate greate summary table
        #if mode != 'estim':
        utils.generate_summary_table(data_name,
                                     exps,
                                     outputfolder,
                                     None,
                                     file_types,
                                     file_types_suffix)

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
                               use_train_valid_for_thr = use_train_valid_for_thr,
                               save_eval_raw_txt = save_eval_raw_txt)

            if mode != 'eval' and mode != 'estim':
                e.prepare()
                e.perform()

            e.evaluate(data_types = data_types)

        # generate greate summary table
        #if mode != 'estim':
        utils.generate_summary_table(data_name,
                                     [exp_name],
                                     outputfolder,
                                     None,
                                     file_types,
                                     file_types_suffix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='path saving test record file')
    args = parser.parse_args()
    record_base_path = args.recording_path

    main(record_base_path)
