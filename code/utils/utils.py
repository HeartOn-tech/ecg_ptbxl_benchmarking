import os
import sys
import re
import glob
import pickle
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import wfdb
import ast
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import warnings
import locale

# EVALUATION STUFF
def generate_results(idxs, y_true, y_pred, thresholds_Fbeta = None, thresholds_Gbeta = None, beta1 = 2, beta2 = 2):
    return evaluate_experiment(y_true[idxs], y_pred[idxs], thresholds_Fbeta, thresholds_Gbeta, beta1, beta2)

def evaluate_experiment(y_true, y_pred, thresholds_Fbeta, thresholds_Gbeta, beta1, beta2):
    results = {}

    # label based metrics
    results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')

    # PhysioNet/CinC Challenges metrics
    if not thresholds_Fbeta is None:
        # binary predictions for Fbeta
        y_pred_binary = apply_thresholds(y_pred, thresholds_Fbeta)
        challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1, beta2)
        results['F_beta_macro'] = challenge_scores['F_beta_macro']

    if not thresholds_Gbeta is None:
        # binary predictions for Gbeta
        y_pred_binary = apply_thresholds(y_pred, thresholds_Gbeta)
        challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1, beta2)
        results['G_beta_macro'] = challenge_scores['G_beta_macro']
    
    df_result = pd.DataFrame(results, index=[0])
    return df_result

def challenge_metrics(y_true, y_pred, beta1 = 2, beta2 = 2, class_weights = None, single = False):
    f_beta = 0
    g_beta = 0
    if single: # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(y_true.sum(axis=1).shape)
    else:
        sample_weights = y_true.sum(axis=1)
    for classi in range(y_true.shape[1]):
        y_truei, y_predi = y_true[:,classi], y_pred[:,classi]
        TP, FP, TN, FN = 0.,0.,0.,0.
        for i in range(len(y_predi)):
            sample_weight = sample_weights[i]
            if y_truei[i]==y_predi[i]==1: 
                TP += 1./sample_weight
            if ((y_predi[i]==1) and (y_truei[i]!=y_predi[i])): 
                FP += 1./sample_weight
            if y_truei[i]==y_predi[i]==0: 
                TN += 1./sample_weight
            if ((y_predi[i]==0) and (y_truei[i]!=y_predi[i])): 
                FN += 1./sample_weight 
        f_beta_i = ((1+beta1**2)*TP)/((1+beta1**2)*TP + FP + (beta1**2)*FN)
        g_beta_i = (TP)/(TP+FP+beta2*FN)

        f_beta += f_beta_i
        g_beta += g_beta_i

    return {'F_beta_macro':f_beta/y_true.shape[1], 'G_beta_macro':g_beta/y_true.shape[1]}

def get_appropriate_bootstrap_samples(y_true, n_bootstraping_samples):
    samples=[]
    while True:
        ridxs = np.random.randint(0, len(y_true), len(y_true))
        if y_true[ridxs].sum(axis=0).min() != 0:
            samples.append(ridxs)
            if len(samples) == n_bootstraping_samples:
                break
    return samples

#def find_optimal_cutoff_threshold(target, predicted):
#    """ 
#    Find the optimal probability cutoff point for a classification model related to event rate
#    """
#    fpr, tpr, threshold = roc_curve(target, predicted)
#    optimal_idx = np.argmax(tpr - fpr)
#    optimal_threshold = threshold[optimal_idx]
#    return optimal_threshold

#def find_optimal_cutoff_thresholds(y_true, y_pred):
#	return [find_optimal_cutoff_threshold(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])]

def find_optimal_cutoff_threshold_for_Fbeta(target, predicted, beta1, n_thresholds = 100):
    thresholds = np.linspace(0.00, 1, n_thresholds)
    scores = [challenge_metrics(target, predicted > t, beta1, single = True)['F_beta_macro'] for t in thresholds]
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx]

def find_optimal_cutoff_thresholds_for_Fbeta(y_true, y_pred, beta1 = 2):
    print("optimize thresholds with respect to F_beta")
    return [find_optimal_cutoff_threshold_for_Fbeta(y_true[:,k][:,np.newaxis], y_pred[:,k][:,np.newaxis], beta1) for k in tqdm(range(y_true.shape[1]))]

def find_optimal_cutoff_threshold_for_Gbeta(target, predicted, beta2, n_thresholds = 100):
    thresholds = np.linspace(0.00, 1, n_thresholds)
    scores = [challenge_metrics(target, predicted > t, beta2, single = True)['G_beta_macro'] for t in thresholds]
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx]

def find_optimal_cutoff_thresholds_for_Gbeta(y_true, y_pred, beta2 = 2):
    print("optimize thresholds with respect to G_beta")
    return [find_optimal_cutoff_threshold_for_Gbeta(y_true[:,k][:,np.newaxis], y_pred[:,k][:,np.newaxis], beta2) for k in tqdm(range(y_true.shape[1]))]

def apply_thresholds(preds, thresholds):
	"""
		apply class-wise thresholds to prediction score in order to get binary format.
		BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
	"""
	tmp = []
	for p in preds:
		tmp_p = (p > thresholds).astype(int)
		if np.sum(tmp_p) == 0:
			tmp_p[np.argmax(p)] = 1
		tmp.append(tmp_p)
	tmp = np.array(tmp)
	return tmp

# DATA PROCESSING STUFF

def load_dataset(data_name, path, sampling_rate, release=False):
    # if path.split('/')[-2] == 'ptbxl':
    if data_name == 'PTBXL':
        # load and convert annotation data
        Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)

    # elif path.split('/')[-2] == 'ICBEB':
    elif data_name == 'ICBEB':
        # load and convert annotation data
        Y = pd.read_csv(os.path.join(path, 'icbeb_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_icbeb(Y, sampling_rate, path)

    return X, Y


def load_raw_data_icbeb(df, sampling_rate, path):

    if sampling_rate == 100:
        fullpath = os.path.join(path, 'raw100.npy')
        if os.path.exists(fullpath):
            data = np.load(fullpath, allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(path, 'records100', str(f))) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(fullpath, 'wb'), protocol=4)
    elif sampling_rate == 500:
        fullpath = os.path.join(path, 'raw500.npy')
        if os.path.exists(fullpath):
            data = np.load(fullpath, allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(path, 'records500', str(f))) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(fullpath, 'wb'), protocol=4)
    return data

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        fullpath = os.path.join(path, 'raw100.npy')
        if os.path.exists(fullpath):
            data = np.load(fullpath, allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(path, f)) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(fullpath, 'wb'), protocol=4)
    elif sampling_rate == 500:
        fullpath = os.path.join(path, 'raw500.npy')
        if os.path.exists(fullpath):
            data = np.load(fullpath, allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(path, f)) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(fullpath, 'wb'), protocol=4)
    return data

# MultiLabelBinarizer: change 'PVC' to 'VPC'
change_class_dict = {'PVC': 'VPC'}

def mlb_change_class(mlb):
    key = next(iter(change_class_dict))
    mlb.classes_[mlb.classes_ == key] = change_class_dict[key]
    mlb.fit([mlb.classes_])
    return mlb

def compute_label_aggregations(df, folder, ctype, change_class):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(os.path.join(folder, 'scp_statements.csv'), index_col=0)
    if change_class:
        aggregation_df.rename(index = change_class_dict, inplace = True)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

def select_data(XX, YY, ctype, min_samples, outputfolder, eval_params, mlb, save):
    # convert multilabel to multi-hot
    mlb_loaded = bool(mlb)
    if mlb_loaded:
        mlb = mlb_change_class(mlb)
    else:
        mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        if not mlb_loaded:
            mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        if not mlb_loaded:
            mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        if not mlb_loaded:
            mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        if not mlb_loaded:
            mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter 
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        if not mlb_loaded:
            mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter 
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        if not mlb_loaded:
            mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    suffix = eval_params['suffix']
    # save LabelBinarizer
    if save and eval_params['save_mlb_file']:
        with open(outputfolder + 'mlb' + suffix + '.pkl', 'wb') as tokenizer:
            pickle.dump(mlb, tokenizer)

    # save data:
    # Y - table based on csv file
    # y - indicator matrix
    # mlb - LabelBinarizer
    if save and eval_params['save_tab_ind_mlb_file']:
        with open(outputfolder + 'tab_ind_mlb' + suffix + '.pkl', 'wb') as tokenizer:
            pickle.dump([YY if suffix else Y, y, mlb], tokenizer)

    return X, Y, y, mlb

def preprocess_signals(X_train, X_validation, X_test, outputfolder, eval_params):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
    
    # Save Standardizer data
    if 'data_set' not in eval_params:
        with open(outputfolder+'standard_scaler.pkl', 'wb') as ss_file:
            pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

# DOCUMENTATION STUFF

class Summary_Table:
    def __init__(self,
                 data_name,
                 folder,
                 eval_params):

        self.data_name = data_name
        self.folder = os.path.normpath(folder)
        self.folder_name = self.folder.split(os.sep)[-1] # output folder name
        self.eval_params = eval_params
        self.suffix = self.eval_params['suffix']
        if 'data_set' in self.eval_params:
            sel_suffix = eval_params['data_set']['sel_suffix']

        if eval_params['use_train_valid_for_thr']: # base: train & valid, estim: test
            self.file_type = 'folds_t_v' + self.suffix + sel_suffix
        else:                                      # base: train, estim: valid, test
            self.file_type = 'folds_t' + self.suffix + sel_suffix

        self.d_types_str = 'results_' + self.file_type + '_' + self.data_name.lower()

        # set system local numeric prefences
        locale.setlocale(locale.LC_NUMERIC, '')

        if self.eval_params['save_to_excel']:
            self.excel_writer = pd.ExcelWriter(os.path.join(self.folder, self.d_types_str + '_' + self.folder_name + '.xlsx'), mode = 'w', engine = 'openpyxl')
            pd.DataFrame().to_excel(self.excel_writer, sheet_name = 'Overall', index = False)
        else:
            self.excel_writer = None

    def __del__(self):
        if self.eval_params['save_to_excel']:
            self.excel_writer.close()

    def load_tables(self, exp, selection):

        if selection is None:
            models = sorted(os.listdir(os.path.join(self.folder, exp, 'models')), key = str.lower)
        else:
            models = sorted(selection, key = str.lower)
        # move 'ensemble' and 'naive' to the end of list
        models.sort(key = lambda s: s == 'ensemble' or s == 'naive')

        data_table = {}

        for model in models:
            res_path = os.path.join(self.folder, exp, 'models', model, 'results', self.file_type + '_results.csv')

            if os.path.isfile(res_path): # file exists
                table_types = pd.read_csv(res_path, index_col = 0) # .applymap(lambda x: locale.format_string("%.4f", x))
            else:
                continue

            for data_type in table_types.index:
                table = table_types.loc[[data_type]]
                table.insert(0, 'Model', model)

                if data_type in data_table:
                    data_table[data_type].append(table)
                else:
                    data_table[data_type] = [table]

        df_cc_table = {}
        for data_type in data_table.keys():
            df_cc_table[data_type] = pd.concat(data_table[data_type], ignore_index = True)

        return df_cc_table

    #def exp_table(exp, folder, selection, data_type):

    #    if selection is None:
    #        models = sorted(os.listdir(os.path.join(folder, exp, 'models')), key = str.lower)
    #    else:
    #        models = sorted(selection, key = str.lower)
    #    # move 'ensemble' and 'naive' to the end of list
    #    models.sort(key = lambda s: s == 'ensemble' or s == 'naive')

    #    data = []
    #    models_out = []
    #    cols = []

    #    for model in models:
    #        res_path = os.path.join(folder, exp, 'models', model, 'results', data_type + '_results.csv') # 'test_results.csv'

    #        if os.path.isfile(res_path): # file exists
    #            me_res = pd.read_csv(res_path, index_col = 0)
    #            #me_res.rename(columns = lambda x: x.replace('macro_auc', 'ROCAUC'), inplace = True)

    #            if cols: # cols are not empty
    #                assert me_res.columns.tolist() != cols, 'Columns in current table are different!'
    #            else: # cols are empty
    #                cols = me_res.columns.tolist()

    #            mcol = []
    #            for col in cols:
    #                point = me_res.loc['point', col]
    #                if set(['upper', 'lower']).issubset(me_res.index):
    #                    #mean = me_res.loc['mean', col]
    #                    unc = max(me_res.loc['upper', col] - point, point - me_res.loc['lower', col])
    #                    mcol.append(locale.format_string("%.4f(%.2d)", (point, int(unc*1000))))
    #                else:
    #                    mcol.append(locale.format_string("%.4f", point))
    #            data.append(mcol)
    #            models_out.append(model)

    #    if data: # data are not empty
    #        data_array = np.array(data)
    #        df = pd.DataFrame(data_array, columns = cols, index = models_out)
    #        df.index.name = 'Model'
    #        #df_index = df[df.index.isin(['naive', 'ensemble'])]
    #        #df_rest = df[~df.index.isin(['naive', 'ensemble'])]
    #        #df = pd.concat([df_rest, df_index])
    #        df.reset_index(level = 0, inplace = True)
    #    else: # data are empty
    #        df = pd.DataFrame()

    #    return df

    # helper output function for markdown tables
    def print_table(self, df_table):

        for data_type in df_table.keys():
            df = df_table[data_type]

            if df.loc[0, 'Model']: # value is not empty
                md_source = '\ndata_type = ' + data_type
                md_source += '\n'
                # print column names
                for i, col in enumerate(df.columns.values):
                    if i == 0:
                        md_source += col.ljust(10)
                    else:
                        md_source += '\t' + col
                #md_source += ''

                for ind in df.index:
                    md_source += '\n'
                    for i, val in enumerate(df.loc[ind]):
                        if i == 0:
                            md_source += val.replace('fastai_', '').ljust(10)
                        else:
                            md_source += '\t' + locale.format_string("%.4f", val)
                    #md_source += ''

                #md_source += '\n'
                print(md_source)

    def generate_summary_table(self,
                               exps,
                               selection):

        #exps = ['exp0', 'exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']
        #df_cc_list = []
        #data_types_out = []

        #for file_type in file_types:
        df_head = pd.DataFrame({'Model': ['folder: ' + self.folder_name + ', '
                                            + 'data_name: ' + self.data_name,
                                            'data_type:']})

        dfs_exps = [] # dataframes of experiments related to whole table

        for exp in exps:
            #df = exp_table(exp, folder, selection, data_type)
            df_table = self.load_tables(exp, selection)
            dfs_types = [] # dataframes of data types related to experiment

            for data_type in df_table.keys():
                #if not df_table[data_type]: # df is not empty
                dfs = [] # dataframes of head and models related to data type
                col_name1 = df_table[data_type].columns[1]
                if self.eval_params['save_to_csv']:
                    df_type = copy.deepcopy(df_table[data_type])
                    df_type[col_name1] = df_type[col_name1].apply(lambda x: str(x).replace('.', ','))
                else:
                    df_type = df_table[data_type]

                if not dfs_exps: # exps_out is empty
                    df_head[col_name1] = ['', data_type]
                    dfs.append(df_head)

                #if not dfs:  # dfs is empty
                dfs.append(pd.DataFrame({'Model': ['', exp],
                                         col_name1: ['', '']}))
                dfs.append(df_type)
                df_cc = pd.concat(dfs, ignore_index = True)

                if dfs_types: # dfs_types is not empty
                    df_cc.drop(df_cc.columns[0], axis = 'columns', inplace = True) # drop 'Model' column
                dfs_types.append(df_cc)

            if dfs_types:
                df_cc_types = pd.concat(dfs_types, axis = 'columns')
                dfs_exps.append(df_cc_types)

                print('\n==================================================================')
                print(self.data_name + ' results, file = ' + self.file_type + ', exp = ' + exp) #, end = ''
                self.print_table(df_table)

        if dfs_exps: # dfs_exps is not empty
            # equalize columns if required
            i_cols_max = max(enumerate(dfs_exps), key = lambda x: x[1].shape[1])[0]
            for i in range(len(dfs_exps)):
                (mi, ni) = dfs_exps[i].shape
                nmax = dfs_exps[i_cols_max].shape[1]
                if ni != nmax:
                    for j in range(ni, nmax):
                        dtype = dfs_exps[i_cols_max].dtypes[j]
                        tmp = np.empty((mi, 1), dtype)
                        if dtype == np.dtype('float'):
                            tmp.fill(np.nan)
                        dfs_exps[i].insert(j, str(j), tmp)
                dfs_exps[i].columns = dfs_exps[i_cols_max].columns

            df_cc_exps = pd.concat(dfs_exps, ignore_index = True)

            if self.eval_params['save_to_csv']:
                df_cc_exps.to_csv(os.path.join(self.folder, self.d_types_str + '_overall' + '.csv'), ';', index = False, decimal = ',')

            if self.eval_params['save_to_excel'] and self.excel_writer:
                df_cc_exps.to_excel(self.excel_writer, sheet_name = 'Overall', index = False)

            #if not dfs: # dfs is empty
            #    continue

            #df_cc = pd.concat(dfs, ignore_index = True)
            #df_cc.to_csv(os.path.join(folder, data_type + '_' + data_name.lower() + '_results' + '.csv'), ';', index = False) #, decimal = ','

            #if df_cc_list: # df_cc_list is not empty
            #    df_cc.drop(df_cc.columns[0], axis = 'columns', inplace = True) # drop 'Model' column
            #df_cc_list.append(df_cc)
            #data_types_out.append(data_type)


        #if df_cc_list: # df_cc_list is not empty
        #    if len(df_cc_list) > 1:
        #        d_types_str = 'united_' + file_types_suffix
        #    else: # len == 1
        #        d_types_str = data_types_out[0]
        #    df_cc_cols = pd.concat(df_cc_list, axis = 'columns')
        #    df_cc_cols.to_csv(os.path.join(folder, d_types_str + '_' + data_name.lower() + '_results' + '.csv'), ';', index = False) #, decimal = ','
