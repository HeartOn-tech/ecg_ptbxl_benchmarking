import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
from sklearn import metrics
from utils import utils
from utils import evaluate

class SCP_Experiment():
    '''
        Experiment on SCP-ECG statements. All experiments based on SCP are performed and evaluated the same way.
    '''

    def __init__(self,
                 data_name,
                 experiment_name,
                 task,
                 datafolder,
                 outputfolder,
                 models,
                 sampling_frequency = 100,
                 min_samples = 0,
                 train_fold = 8,
                 val_fold = 9,
                 test_fold = 10,
                 folds_type = 'strat',
                 mode = 'predict',
                 eval_params = {},
                 excel_writer = None):

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
        self.eval_params = eval_params # evaluation module additional options
        if self.eval_params['save_to_excel']:
            self.excel_writer = excel_writer # pd.ExcelWriter object
        else:
            self.excel_writer = None

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

        if 'data_set' in self.eval_params:
            data_set = self.eval_params['data_set']
            data_name = data_set['data_name']
            task = data_set['task']
            datafolder = data_set['datafolder']
        else:
            data_name = self.data_name
            task = self.task
            datafolder = self.datafolder

        self.suffix = self.eval_params['suffix']

        # Load data
        self.raw_data, self.raw_labels = utils.load_dataset(data_name, datafolder, self.sampling_frequency)

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, datafolder, task, bool(self.suffix))

        if 'data_set' in self.eval_params:
            mlb_path = os.path.join(self.outputfolder, self.experiment_name, 'data', 'tab_ind_mlb.pkl')
            if os.path.isfile(mlb_path): # file exists
                with open(mlb_path, 'rb') as tokenizer:
                    _, _, mlb = pickle.load(tokenizer)
                    self.n_classes = len(mlb.classes_)
            else:
                raise Exception('mlb file does not exist!')
        else:
            mlb = None

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.raw_data, self.labels, task, self.min_samples, self.outputfolder + self.experiment_name + '/data/', self.eval_params, mlb, not bool(self.suffix))

        if 'data_set' in self.eval_params:
            self.labels_ds = utils.compute_label_aggregations(self.raw_labels, datafolder, self.task, bool(self.suffix))
            utils.select_data(self.raw_data, self.labels_ds, self.task, self.min_samples, self.outputfolder + self.experiment_name + '/data/', self.eval_params, mlb, True)

        self.input_shape = self.data[0].shape

        #if self.mode == 'eval':
        #    return
        
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
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder + self.experiment_name + '/data/', self.eval_params)

        # save train and test labels
        if not hasattr(self, 'n_classes'):
            self.n_classes = self.Y.shape[1]

        self.y_train.dump(self.outputfolder + self.experiment_name+ '/data/y_train' + self.suffix + '.npy')
        self.y_val.dump(self.outputfolder + self.experiment_name+ '/data/y_val' + self.suffix + '.npy')
        self.y_test.dump(self.outputfolder + self.experiment_name+ '/data/y_test' + self.suffix + '.npy')

        modelname = 'naive'
        # create most naive predictions via simple mean in training
        mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        if not os.path.exists(mpath+'results/'):
            os.makedirs(mpath+'results/')

        mean_y = np.mean(self.y_train, axis=0)
        np.array([mean_y]*len(self.y_train)).dump(mpath + 'y_train_pred' + self.suffix + '.npy')
        np.array([mean_y]*len(self.y_val)).dump(mpath + 'y_val_pred' + self.suffix + '.npy')
        np.array([mean_y]*len(self.y_test)).dump(mpath + 'y_test_pred' + self.suffix + '.npy')

    def perform(self):

        # predict and dump
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

            if self.mode == 'finetune':
                pretrained = True
                pretrainedfolder = mpath
                n_classes_pretrained = self.n_classes
            else:
                pretrained = False
                pretrainedfolder = None
                n_classes_pretrained=None

            # load respective model
            if modeltype == 'WAVELET':
                from models.wavelet import WaveletModel
                model = WaveletModel(modelname, self.n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "fastai_model":
                from models.fastai_model import fastai_model
                #modelparams['epochs'] = 5
                #modelparams['epochs_finetuning'] = 5
                model = fastai_model(modelname, self.n_classes, self.sampling_frequency, mpath, self.input_shape, pretrained, pretrainedfolder = pretrainedfolder, n_classes_pretrained = n_classes_pretrained, **modelparams)
            elif modeltype == "YOUR_MODEL_TYPE":
                # YOUR MODEL GOES HERE!
                from models.your_model import YourModel
                model = YourModel(modelname, self.n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            else:
                assert(True)
                break
    
            # fit model
            if self.mode in ['fit', 'finetune']:
                model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

            if 'data_set' in self.eval_params:
                dataoutputfolder = self.outputfolder + self.eval_params['data_set']['exp_name'] + '/models/' + modelname + '/'
            else:
                dataoutputfolder = None

            model.predict(self.X_train, 'train', dataoutputfolder).dump(mpath + 'y_train_pred' + self.suffix + '.npy')
            model.predict(self.X_val, 'val', dataoutputfolder).dump(mpath + 'y_val_pred' + self.suffix + '.npy')
            model.predict(self.X_test, 'test', dataoutputfolder).dump(mpath + 'y_test_pred' + self.suffix + '.npy')

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
                ensemble_train.append(np.load(mpath+'y_train_pred' + self.suffix + '.npy', allow_pickle=True))
                ensemble_val.append(np.load(mpath+'y_val_pred' + self.suffix + '.npy', allow_pickle=True))
                ensemble_test.append(np.load(mpath+'y_test_pred' + self.suffix + '.npy', allow_pickle=True))
        # dump mean predictions
        np.array(ensemble_train).mean(axis=0).dump(ensemblepath + 'y_train_pred' + self.suffix + '.npy')
        np.array(ensemble_val).mean(axis=0).dump(ensemblepath + 'y_val_pred' + self.suffix + '.npy')
        np.array(ensemble_test).mean(axis=0).dump(ensemblepath + 'y_test_pred' + self.suffix + '.npy')

    def evaluate(self,
                 n_bootstraping_samples = 100,
                 n_jobs = 20,
                 bootstrap_eval = False,
                 dumped_bootstraps = True):

        # if bootstrapping then generate appropriate samples for each
        #if bootstrap_eval:
        #    sample_inds = {}
        #    if not dumped_bootstraps:
        #        for data_type in data_types:
        #            sample_inds[data_type] = np.array(utils.get_appropriate_bootstrap_samples(y_labels[data_type], n_bootstraping_samples))
        #            # store samples for future evaluations
        #            name = utils.data_type_to_name(data_type)
        #            sample_inds[data_type].dump(self.outputfolder + self.experiment_name + '/' + name + '_bootstrap_ids.npy')
        #    else:
        #        for data_type in data_types:
        #            name = utils.data_type_to_name(data_type)
        #            sample_inds[data_type] = np.load(self.outputfolder + self.experiment_name + '/' + name + '_bootstrap_ids.npy', allow_pickle=True)
        #else:
        #    sample_inds['train'] = np.array([range(len(y_labels['train']))]) # y_train
        #    sample_inds['val'] = np.array([range(len(y_labels['val']))]) # y_val
        #    sample_inds['test'] = np.array([range(len(y_labels['test']))]) # y_test

        # create Evaluation object
        data_types = self.eval_params['data_types']
        eval_obj = evaluate.Evaluation(self.outputfolder,
                                       self.data_name,
                                       self.experiment_name,
                                       self.task,
                                       {data_types[0]: self.train_fold, data_types[1]: self.val_fold, data_types[2]: self.test_fold},
                                       self.eval_params,
                                       self.excel_writer)

        # calc evaluation results for base data with threshold evaluation,
        # evaluation results for test (or valid and test) data
        # and for additional train folds
        if eval_obj: # folder 'data' exists
            eval_obj.challenge_metrics_models()

        # effectiveness evaluation
        #beta1 = 2 # Fbeta parameter
        #beta2 = 2 # Gbeta parameter
        #if self.data_name == 'ICBEB':
        # compute classwise thresholds such that recall-focused Fbeta is optimized
        #thresholds_Fbeta = utils.find_optimal_cutoff_thresholds_for_Fbeta(y_labels['train'], y_preds['train'], beta1) # y_train, y_train_pred
        # compute classwise thresholds such that recall-focused Gbeta is optimized
        #thresholds_Gbeta = utils.find_optimal_cutoff_thresholds_for_Gbeta(y_labels['train'], y_preds['train'], beta2) # y_train, y_train_pred
        #else:
        #    thresholds = None

        #pool = multiprocessing.Pool(n_jobs)

        #for key in y_labels.keys():
        #    print('generate_results(), data_type = ', key)
        #    # all samples
        #    df_point = utils.generate_results(range(len(y_labels[key])), y_labels[key], y_preds[key], # range(len(y_test)), y_test, y_test_pred, thresholds
        #        thresholds_Fbeta, thresholds_Gbeta, beta1, beta2)

        #    if bootstrap_eval:
        #        df = pd.concat(pool.starmap(utils.generate_results, zip(sample_inds[key], repeat(y_labels[key]), repeat(y_preds[key]), # test_samples, repeat(y_test), repeat(y_test_pred)
        #            repeat(thresholds_Fbeta), repeat(thresholds_Fbeta), repeat(beta1), repeat(beta2))))

        #        df_result = pd.DataFrame(
        #            np.array([
        #                df_point.values[0],
        #                df.mean().values,
        #                df.quantile(0.05).values,
        #                df.quantile(0.95).values]), 
        #            columns = df.columns, 
        #            index = ['point', 'mean', 'lower', 'upper'])
        #    else:
        #        df_result = pd.DataFrame(
        #            df_point.values,
        #            columns = df_point.columns,
        #            index = ['point'])

        #    # dump results
        #    df_result.to_csv(rpath + key + '_results' + '.csv')

        #pool.close()

        #tr_df_result.to_csv(rpath+'tr_results.csv')
        #val_df_result.to_csv(rpath+'val_results.csv')
        #te_df_result.to_csv(rpath+'te_results.csv')
