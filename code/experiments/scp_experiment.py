from utils import utils
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
from sklearn import metrics

class SCP_Experiment():
    '''
        Experiment on SCP-ECG statements. All experiments based on SCP are performed and evaluated the same way.
    '''

    def __init__(self, data_name, experiment_name, task, datafolder, outputfolder, models, sampling_frequency=100, min_samples=0, train_fold=8, val_fold=9, test_fold=10, folds_type='strat', mode = 'predict'):
        self.models = models
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds_type = folds_type
        self.experiment_name = experiment_name
        self.outputfolder = outputfolder
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency
        self.data_name = data_name
        self.mode = mode

        # create folder structure if needed
        exp_folder = os.path.join(self.outputfolder, self.experiment_name)
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
        if not os.path.exists(os.path.join(exp_folder, 'results')):
            os.makedirs(os.path.join(exp_folder, 'results'))
        if not os.path.exists(os.path.join(exp_folder, 'models')):
            os.makedirs(os.path.join(exp_folder, 'models'))
        if not os.path.exists(os.path.join(exp_folder, 'data')):
            os.makedirs(os.path.join(exp_folder, 'data'))

    def prepare(self):
        # Load data
        self.data, self.raw_labels = utils.load_dataset(self.data_name, self.datafolder, self.sampling_frequency)

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')
        self.input_shape = self.data[0].shape
        
        # 10th fold for testing (9th for now)
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        self.n_classes = self.y_train.shape[1]

        # save train and test labels
        self.y_train.dump(self.outputfolder + self.experiment_name+ '/data/y_train.npy')
        self.y_val.dump(self.outputfolder + self.experiment_name+ '/data/y_val.npy')
        self.y_test.dump(self.outputfolder + self.experiment_name+ '/data/y_test.npy')

        modelname = 'naive'
        # create most naive predictions via simple mean in training
        mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        if not os.path.exists(mpath+'results/'):
            os.makedirs(mpath+'results/')

        mean_y = np.mean(self.y_train, axis=0)
        np.array([mean_y]*len(self.y_train)).dump(mpath + 'y_train_pred.npy')
        np.array([mean_y]*len(self.y_test)).dump(mpath + 'y_test_pred.npy')
        np.array([mean_y]*len(self.y_val)).dump(mpath + 'y_val_pred.npy')

    def perform(self):

        for model_description in self.models:
            modelname = model_description['modelname']
            modeltype = model_description['modeltype']
            modelparams = model_description['parameters']

            mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
            # create folder for model outputs
            if not os.path.exists(mpath):
                os.makedirs(mpath)
            if not os.path.exists(mpath+'results/'):
                os.makedirs(mpath+'results/')

            n_classes = self.Y.shape[1]

            if self.mode == 'finetune':
                pretrained = True
                pretrainedfolder = mpath
                n_classes_pretrained = n_classes
            else:
                pretrained = False
                pretrainedfolder = None
                n_classes_pretrained=None

            # load respective model
            if modeltype == 'WAVELET':
                from models.wavelet import WaveletModel
                model = WaveletModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "fastai_model":
                from models.fastai_model import fastai_model
                #modelparams['epochs'] = 5
                #modelparams['epochs_finetuning'] = 5
                model = fastai_model(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, pretrained, pretrainedfolder = pretrainedfolder, n_classes_pretrained = n_classes_pretrained, **modelparams)
            elif modeltype == "YOUR_MODEL_TYPE":
                # YOUR MODEL GOES HERE!
                from models.your_model import YourModel
                model = YourModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            else:
                assert(True)
                break

            # fit model
            if self.mode in ['fit', 'finetune']:
                model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

            # predict and dump
            model.predict(self.X_train, 'train').dump(mpath+'y_train_pred.npy')
            model.predict(self.X_val, 'val').dump(mpath+'y_val_pred.npy')
            model.predict(self.X_test, 'test').dump(mpath+'y_test_pred.npy')

        modelname = 'ensemble'
        # create ensemble predictions via simple mean across model predictions (except naive predictions)
        ensemblepath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(ensemblepath):
            os.makedirs(ensemblepath)
        if not os.path.exists(ensemblepath+'results/'):
            os.makedirs(ensemblepath+'results/')
        # load all predictions
        ensemble_train, ensemble_val, ensemble_test = [],[],[]
        for model_description in os.listdir(self.outputfolder+self.experiment_name+'/models/'):
            if not model_description in ['ensemble', 'naive']:
                mpath = self.outputfolder+self.experiment_name+'/models/'+model_description+'/'
                ensemble_train.append(np.load(mpath+'y_train_pred.npy', allow_pickle=True))
                ensemble_val.append(np.load(mpath+'y_val_pred.npy', allow_pickle=True))
                ensemble_test.append(np.load(mpath+'y_test_pred.npy', allow_pickle=True))
        # dump mean predictions
        np.array(ensemble_train).mean(axis=0).dump(ensemblepath + 'y_train_pred.npy')
        np.array(ensemble_val).mean(axis=0).dump(ensemblepath + 'y_val_pred.npy')
        np.array(ensemble_test).mean(axis=0).dump(ensemblepath + 'y_test_pred.npy')

    def evaluate(self, n_bootstraping_samples=100, n_jobs=20, bootstrap_eval=False, dumped_bootstraps=True, data_types = ['test']):
        # get labels
        y_labels = {}
        data_types_ext = [data_type for data_type in data_types]
        if not 'train' in data_types_ext:
            data_types_ext.insert(0, 'train')

        for data_type in data_types_ext:
            name = utils.data_type_to_name(data_type)
            y_labels[data_type] = np.load(self.outputfolder + self.experiment_name + '/data/y_' + name + '.npy', allow_pickle=True)

        # if bootstrapping then generate appropriate samples for each
        if bootstrap_eval:
            sample_inds = {}
            if not dumped_bootstraps:
                for data_type in data_types:
                    sample_inds[data_type] = np.array(utils.get_appropriate_bootstrap_samples(y_labels[data_type], n_bootstraping_samples))
                    # store samples for future evaluations
                    name = utils.data_type_to_name(data_type)
                    sample_inds[data_type].dump(self.outputfolder + self.experiment_name + '/' + name + '_bootstrap_ids.npy')
            else:
                for data_type in data_types:
                    name = utils.data_type_to_name(data_type)
                    sample_inds[data_type] = np.load(self.outputfolder + self.experiment_name + '/' + name + '_bootstrap_ids.npy', allow_pickle=True)
        #else:
        #    sample_inds['train'] = np.array([range(len(y_labels['train']))]) # y_train
        #    sample_inds['val'] = np.array([range(len(y_labels['val']))]) # y_val
        #    sample_inds['test'] = np.array([range(len(y_labels['test']))]) # y_test

        # iterate over all models fitted so far
        label_inds = range(y_labels['train'].shape[1])
        roc_curve_cols = ['threshold', 'fpr', 'tpr']
        pr_curve_cols = ['threshold', 'precision', 'recall']
        det_curve_cols = ['threshold', 'fpr', 'fnr']

        for m in sorted(os.listdir(self.outputfolder + self.experiment_name + '/models')):
            print('evaluate(): data_name:', self.data_name, ', exp:', self.experiment_name, ', model:', m)
            mpath = self.outputfolder + self.experiment_name + '/models/' + m + '/'
            rpath = self.outputfolder + self.experiment_name + '/models/' + m + '/results/'

            # load predictions
            y_preds = {}
            for data_type in data_types_ext:
                name = utils.data_type_to_name(data_type)
                y_preds[data_type] = np.load(mpath + 'y_' + name + '_pred.npy', allow_pickle=True)

            if self.mode == 'estim':
                # thresholds estimation
                conf_m_dfs = [] # conf matricies
                roc_dfs = [] # roc dataframes
                pr_dfs = [] #precision-recall dataframes
                det_dfs = [] #det dataframes
                for l in label_inds: # cycle by labels (columns)
                    y_labels_col = y_labels['train'][:, l]
                    y_preds_col = y_preds['train'][:, l]
                    # tp, fp, tn, fn
                    conf_m, y_pred_uniq = utils.conf_matrix(y_labels_col, y_preds_col)
                    df_conf_m = pd.DataFrame(conf_m, columns = ['tp', 'fp', 'tn', 'fn'])
                    df_conf_m.insert(0, 'threshold', y_pred_uniq)
                    df_conf_m['FPR'] = conf_m[:, 1] / (conf_m[:, 1] + conf_m[:, 2]) # fpr = fp / (fp + tn)
                    df_conf_m['FNR'] = conf_m[:, 3] / (conf_m[:, 0] + conf_m[:, 3]) # fnr = fn / (fn + tp)
                    df_conf_m['PPV'] = conf_m[:, 0] / (conf_m[:, 0] + conf_m[:, 1]) # ppv = tp / (tp + fp)

                    # roc_curve
                    roc_fpr, roc_tpr, roc_thresholds = metrics.roc_curve(y_labels_col, y_preds_col)
                    n_roc_thrs = roc_thresholds.shape[0]
                    roc_res = np.zeros((n_roc_thrs, len(roc_curve_cols)))
                    roc_res[:, 0] = roc_thresholds
                    roc_res[:, 1] = roc_fpr[0:n_roc_thrs]
                    roc_res[:, 2] = roc_tpr[0:n_roc_thrs]
                    # precision_recall_curve
                    pr_precision, pr_recall, pr_thresholds = metrics.precision_recall_curve(y_labels_col, y_preds_col)
                    n_pr_thrs = pr_thresholds.shape[0]
                    pr_res = np.zeros((n_pr_thrs, len(pr_curve_cols)))
                    pr_res[:, 0] = pr_thresholds
                    pr_res[:, 1] = pr_precision[0:n_pr_thrs]
                    pr_res[:, 2] = pr_recall[0:n_pr_thrs]
                    # det_curve
                    det_fpr, det_fnr, det_thresholds = metrics.det_curve(y_labels_col, y_preds_col)
                    n_det_thrs = det_thresholds.shape[0]
                    det_res = np.zeros((n_det_thrs, len(det_curve_cols)))
                    det_res[:, 0] = det_thresholds
                    det_res[:, 1] = det_fpr[0:n_det_thrs]
                    det_res[:, 2] = det_fnr[0:n_det_thrs]

                    conf_m_dfs.append(df_conf_m)
                    roc_dfs.append(pd.DataFrame(roc_res, columns = roc_curve_cols))
                    pr_dfs.append(pd.DataFrame(pr_res, columns = pr_curve_cols))
                    det_dfs.append(pd.DataFrame(det_res, columns = det_curve_cols))

                df_conf_m_res = pd.concat(conf_m_dfs, keys = label_inds, names = ['label', 'i'])
                df_conf_m_res.to_csv(rpath + 'train' + '_conf_mat' + '.csv', ';', decimal = ',')
                df_roc_res = pd.concat(roc_dfs, keys = label_inds, names = ['label', 'i'])
                df_roc_res.to_csv(rpath + 'train' + '_roc_curves' + '.csv', ';', decimal = ',')
                df_pr_res = pd.concat(pr_dfs, keys = label_inds, names = ['label', 'i'])
                df_pr_res.to_csv(rpath + 'train' + '_pr_curves' + '.csv', ';', decimal = ',')
                df_det_res = pd.concat(det_dfs, keys = label_inds, names = ['label', 'i'])
                df_det_res.to_csv(rpath + 'train' + '_det_curves' + '.csv', ';', decimal = ',')

            else:
                # effectiveness evaluation
                beta1 = 2 # Fbeta parameter
                beta2 = 2 # Gbeta parameter
                #if self.data_name == 'ICBEB':
                # compute classwise thresholds such that recall-focused Fbeta is optimized
                thresholds_Fbeta = utils.find_optimal_cutoff_thresholds_for_Fbeta(y_labels['train'], y_preds['train'], beta1) # y_train, y_train_pred
                # compute classwise thresholds such that recall-focused Gbeta is optimized
                #thresholds_Gbeta = utils.find_optimal_cutoff_thresholds_for_Gbeta(y_labels['train'], y_preds['train'], beta2) # y_train, y_train_pred
                #else:
                #    thresholds = None

                pool = multiprocessing.Pool(n_jobs)

                for key in y_labels.keys():
                    print('generate_results(), data_type = ', key)
                    # all samples
                    df_point = utils.generate_results(range(len(y_labels[key])), y_labels[key], y_preds[key], # range(len(y_test)), y_test, y_test_pred, thresholds
                        thresholds_Fbeta, thresholds_Gbeta, beta1, beta2)

                    if bootstrap_eval:
                        df = pd.concat(pool.starmap(utils.generate_results, zip(sample_inds[key], repeat(y_labels[key]), repeat(y_preds[key]), # test_samples, repeat(y_test), repeat(y_test_pred)
                            repeat(thresholds_Fbeta), repeat(thresholds_Fbeta), repeat(beta1), repeat(beta2))))

                        df_result = pd.DataFrame(
                            np.array([
                                df_point.values[0],
                                df.mean().values,
                                df.quantile(0.05).values,
                                df.quantile(0.95).values]), 
                            columns = df.columns, 
                            index = ['point', 'mean', 'lower', 'upper'])
                    else:
                        df_result = pd.DataFrame(
                            df_point.values,
                            columns = df_point.columns,
                            index = ['point'])

                    # dump results
                    df_result.to_csv(rpath + key + '_results' + '.csv')

                pool.close()

                #tr_df_result.to_csv(rpath+'tr_results.csv')
                #val_df_result.to_csv(rpath+'val_results.csv')
                #te_df_result.to_csv(rpath+'te_results.csv')
