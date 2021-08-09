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
    mode = 'predict' # mode: 'predict' - только выполнение предсказания, иначе выполнение обучения - fit(), 'finetune' - дообучение моделей (только fastai-модели)

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
            e = SCP_Experiment(data_name, exp_name, task, datafolder_ptbxl, outputfolder, models, mode = mode)
            e.prepare()
            e.perform()
            e.evaluate()

        #generate greate summary table
        utils.generate_ptbxl_summary_table(exps)

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################
    use_ICBEB = True
    if use_ICBEB:
        data_name = 'ICBEB'
        exp_name = 'exp_ICBEB'
        task = 'all'

        e = SCP_Experiment(data_name, exp_name, task, datafolder_icbeb, outputfolder, models, mode = mode)
        e.prepare()
        e.perform()
        e.evaluate()

        # generate greate summary table
        utils.ICBEBE_table(exp_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='path saving test record file')
    args = parser.parse_args()
    record_base_path = args.recording_path

    main(record_base_path)
