import os
import argparse
from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *


def main(record_base_path):
    datafolder_ptbxl = os.path.join(
        record_base_path,
        'ptbxl')  # '../data/ptbxl/'
    datafolder_icbeb = os.path.join(
        record_base_path,
        'ICBEB')  # '../data/ICBEB/'
    outputfolder = '../output/'
    mode = 'estim'
    # 'results' - только таблица результатов
    # 'estim' - только оценка эффективности моделей в зависимости от порога
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

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################
    data_types = ['train', 'valid', 'test']

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
                e = SCP_Experiment(
                    data_name,
                    exp_name,
                    task,
                    datafolder_ptbxl,
                    outputfolder,
                    models,
                    mode=mode)
                if mode != 'eval' and mode != 'estim':
                    e.prepare()
                    e.perform()
                e.evaluate(data_types=data_types)

        # generate greate summary table
        if mode != 'estim':
            for data_type in data_types:
                utils.generate_ptbxl_summary_table(
                    exps, outputfolder, None, data_type)

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################
    use_ICBEB = True
    if use_ICBEB:
        data_name = 'ICBEB'
        exp_name = 'exp_ICBEB'
        task = 'all'

        if mode != 'results':
            e = SCP_Experiment(
                data_name,
                exp_name,
                task,
                datafolder_icbeb,
                outputfolder,
                models,
                mode=mode)
            if mode != 'eval' and mode != 'estim':
                e.prepare()
                e.perform()
            e.evaluate(data_types=data_types)

        # generate greate summary table
        if mode != 'estim':
            for data_type in data_types:
                utils.ICBEBE_table(exp_name, outputfolder, None, data_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='path saving test record file')
    args = parser.parse_args()
    record_base_path = args.recording_path

    main(record_base_path)
