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

# calc conf matrix for unique sorted thresholds
def conf_matrix(y_true, y_pred):
    inds = np.argsort(y_pred)
    y_pred_uniq = np.unique(y_pred[inds]) # return_index = True
    #inds = range(len(y_pred_uniq))
    out = np.zeros((len(y_pred_uniq), 4), dtype = int) # array by y_th and TP, FP, TN, FN

    # TP, FP
    first_it = True
    i_prev = inds[-1]
    j = len(y_pred_uniq) # out[j]
    TP_tmp = 0
    FP_tmp = 0

    for i in inds[::-1]: # y_true[i], y_pred[i]

        if y_true[i]: # y_true[i] = 1
            TP_tmp += 1 # TP
        else: # y_true[i] = 0
            FP_tmp += 1 # FP

        if first_it or y_pred[i] != y_pred[i_prev]: # new out elem
            j -= 1
            if first_it:
                first_it = False

        out[j][0] = TP_tmp
        out[j][1] = FP_tmp
        # print('j = ', j, 'y_pred[i] = ', y_pred[i], ', i = ', i, 'i_prev = ', i_prev)

        i_prev = i

    # TN, FN
    i_prev = inds[0]
    j = 0 # out[j]
    TN_tmp = 0
    FN_tmp = 0

    for i in inds[1:]: # y_true[i], y_pred[i]

        if not y_true[i_prev]: # y_true[i-1] = 0
            TN_tmp += 1 # TN
        else:
            FN_tmp += 1 # FN

        if y_pred[i] != y_pred[i_prev]: # new out elem
            j += 1
            out[j][2] = TN_tmp
            out[j][3] = FN_tmp
        # print('j = ', j, 'y_pred[i] = ', y_pred[i], ', i = ', i, 'i_prev = ', i_prev)

        i_prev = i

    return out, y_pred_uniq

# build graph
def build_graph(pdf, y_pred, fpr_fnr_etc, fpr_fnr_labels, text):
    matplotlib.use('agg') # due to this memory leakage problem was solved

    mm = 1.0 / 25.4  # milimeters in inches
    fig, (ax_txt, ax1, ax2) = plt.subplots(nrows = 3, ncols = 1, figsize = (160*mm, 296.97*mm), gridspec_kw = {'height_ratios': [1, 10, 10]})

    # text graph
    ax_txt.axis('off')
    ax_txt.text(0.5, 0.0, text, size = 12, ha = 'center')

    y_up_lim = np.max(y_pred)
    if y_up_lim > 1.0:
        y_up_lim = 1.0
    # main graph
    for i in range(len(fpr_fnr_labels)):
        ax1.plot(y_pred, fpr_fnr_etc[:, i], label = fpr_fnr_labels[i], linewidth = 1.0)
    ax1.set_xlim(0.0, y_up_lim)
    ax1.set_ylim(0.0, 1.0)
    ax1.set(xlabel = 'threshold', ylabel = 'FPR, FNR, etc', title = 'FPR, FNR, etc')
    ax1.legend(loc = 'center right')
    ticks = np.linspace(0.0, 1.0, 11)
    if int(np.ceil(10.0 * y_up_lim)) == 10:
        ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    #ax1.minorticks_on()
    ax1.grid(True, which = 'major', linestyle = '--')
    #ax1.grid(True, which = 'minor', linestyle = '--')

    # ROC graph
    ax2.plot(fpr_fnr_etc[:, 0], fpr_fnr_etc[:, 2], label = 'ROC', linewidth = 1.0)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set(xlabel = fpr_fnr_labels[0], ylabel = fpr_fnr_labels[2], title = 'ROC')
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.grid(True, which = 'major', linestyle = '--')

    plt.tight_layout()
    pdf.savefig()
    fig.clf()
    plt.close()

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
    if data_name == 'ptbxl':
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

def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(os.path.join(folder, 'scp_statements.csv'), index_col=0)

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

def select_data(XX,YY, ctype, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
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
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # save LabelBinarizer
    with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb

def preprocess_signals(X_train, X_validation, X_test, outputfolder):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
    
    # Save Standardizer data
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

def data_type_to_name(data_type):
    if data_type == 'valid':
        name = 'val'
    else:
        name = data_type
    return name

# DOCUMENTATION STUFF
def exp_table(exp, folder, selection, data_type):
    cols = ['macro_auc', 'F_beta_macro', 'G_beta_macro']

    if selection is None:
        #models = [m.split('_pretrained')[0] for m in glob.glob(os.path.join(folder, exp, 'models'))] # m.split('/')[-1].split('_pretrained')[0] '/models/*'
        models = os.listdir(os.path.join(folder, exp, 'models'))
    else:
        models = selection
        #for s in selection:
        #    models.append(s)

    data = []
    for model in models:
        res_path = os.path.join(folder, exp, 'models', model, 'results', data_type + '_results.csv') # 'test_results.csv'
        if os.path.isfile(res_path):
            me_res = pd.read_csv(res_path, index_col=0)
            mcol=[]
            for col in cols:
                if col in me_res.columns:
                    point = me_res.loc['point'][col]
                    if set(['upper', 'lower']).issubset(me_res.index):
                        #mean = me_res.loc['mean'][col]
                        unc = max(me_res.loc['upper'][col] - point, point - me_res.loc['lower'][col])
                        mcol.append("%.3f(%.2d)" %(np.round(point, 3), int(unc*1000)))
                    else:
                        mcol.append("%.3f" %(np.round(point, 3)))
            data.append(mcol)

    data_array = np.array(data)

    df = pd.DataFrame(data_array, columns = cols, index = models)
    df_index = df[df.index.isin(['naive', 'ensemble'])]
    df_rest = df[~df.index.isin(['naive', 'ensemble'])]
    df = pd.concat([df_rest, df_index])
    return df

def generate_ptbxl_summary_table(exps, folder, selection = None, data_type = 'test'):
    #exps = ['exp0', 'exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']
    #metric1 = 'macro_auc'

    dfs = [] # dataframes related to experiments
    for exp in exps:
        dfs.append(exp_table(exp, folder, selection, data_type))

    df = pd.concat(dfs, keys = exps)
    df.to_csv(os.path.join(folder, data_type + '_results_ptbxl.csv'))

    # helper output function for markdown tables
    print('PTB_XL results, data_type = ' + data_type, end = '')
    #df_rest = df[~df.index.isin(['naive', 'ensemble'])]
    #df_rest = df_rest.sort_values('macro_auc', ascending=False)
    df_rest = df
    cols = [col for col in df_rest.columns if col]
    #our_work = 'https://arxiv.org/abs/2004.13701'
    #our_repo = 'https://github.com/helme/ecg_ptbxl_benchmarking/'
    md_head = '| Model ' # | AUC &darr; |  F_beta=2 | G_beta=2 | paper/source | code | \n'
    for col in cols:
        md_head += '| ' + col
    md_head += ' |' # '|---:|:---|:---|:---|:---|:---|\n'

    md_source = ''
    for exp in exps:
        md_source += '\nexp = ' + exp
        md_source += '\n' + md_head
        for i, row in enumerate(df_rest.loc[exp, cols].values):
            md_source += '\n| ' + df_rest.loc[exp].index[i].replace('fastai_', '') + ' | ' + row[0] + ' | ' + row[1] + ' | ' + row[2] # + ' | [our work]('+our_work+') | [this repo]('+our_repo+')| \n'

    md_source += '\n'
    print(md_source)

def ICBEBE_table(exp, folder, selection = None, data_type = 'test'):
    df = exp_table(exp, folder, selection, data_type)
    df.to_csv(os.path.join(folder, data_type + '_results_icbeb.csv'))

    # helper output function for markdown tables
    print('ICBEB results, data_type = ' + data_type)
    #df_rest = df[~df.index.isin(['naive', 'ensemble'])]
    #df_rest = df_rest.sort_values('macro_auc', ascending=False)
    df_rest = df
    cols = [col for col in df_rest.columns if col]
    #our_work = 'https://arxiv.org/abs/2004.13701'
    #our_repo = 'https://github.com/helme/ecg_ptbxl_benchmarking/'
    md_source = '| Model ' # | AUC &darr; |  F_beta=2 | G_beta=2 | paper/source | code | \n'
    for col in cols:
        md_source += '| ' + col
    md_source += ' |' # '|---:|:---|:---|:---|:---|:---|\n'

    for i, row in enumerate(df_rest[cols].values):
        md_source += '\n| ' + df_rest.index[i].replace('fastai_', '') + ' | ' + row[0] + ' | ' + row[1] + ' | ' + row[2] # + ' | [our work]('+our_work+') | [this repo]('+our_repo+')| \n'

    md_source += '\n'
    print(md_source)
