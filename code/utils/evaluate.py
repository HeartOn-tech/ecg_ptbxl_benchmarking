import os
import sys
import re
import glob
import pickle
import copy
import pandas as pd
import numpy as np
#import multiprocessing
#from tqdm import tqdm
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import utils

# valid to val
def data_type_to_name(data_type):
    if data_type == 'valid':
        name = 'val'
    else:
        name = data_type
    return name

# evaluation class
class Evaluation:

    def __init__(self,
                 outputfolder,
                 data_name,
                 experiment_name,
                 task,
                 data_folds,
                 eval_params,
                 excel_writer):
        # options
        self.eval_params = eval_params
        self.data_types_ext = self.eval_params['data_types'] # data types for processing
        self.N_thrs = self.eval_params['N_thrs'] # number of threshold criterion
        self.use_train_valid_for_thr = self.eval_params['use_train_valid_for_thr'] # True: use train and valid for threshold evaluation, False: use train only for threshold evaluation

        # graph parameters
        self.mm = 1.0 / 25.4  # milimeters in inches
        self.figsize = (160.0 * self.mm, 296.97 * self.mm)

        # file paths
        self.output_folder = outputfolder
        norm_output_folder = os.path.normpath(self.output_folder)
        self.output_folder_name = norm_output_folder.split(os.sep)[-1] # output folder name
        self.data_name = data_name
        self.experiment_name = experiment_name
        self.task = task
        self.data_folder = os.path.join(norm_output_folder, self.experiment_name, 'data')
        self.suffix = self.eval_params['suffix']
        # output text
        self.out_text = ['folder: ' + self.output_folder_name,
                    'data_name: ' + data_name,
                    'exp: ' + self.experiment_name]
        self.data_exp_text = self.out_text[1] + ', ' + self.out_text[2]
        self.evaluate_text = 'evaluate(): ' + self.data_exp_text

        if not os.path.exists(self.data_folder): # folder does not exist
            print(self.evaluate_text, '\nError: folder', self.data_folder, 'does not exist!')
            self.state = False # False state of object
            return

        self.results_folder = os.path.join(norm_output_folder, self.experiment_name, 'results')
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        self.data_folds = data_folds

        # create column names
        self.dtype_list = [('label_i', int),
                           ('label', str),
                           ('Np', int),
                           ('Nn', int),
                           ('ROCAUC', float)]

        for k in range(self.N_thrs): # loop for N_thr
            str_num = str(k + 1)
            self.dtype_list.append(('thr' + str_num, float))
            self.dtype_list.append(('FPR' + str_num, float))
            self.dtype_list.append(('FNR' + str_num, float))

        self.dtypes = np.dtype(self.dtype_list)

        self.fpr_fnr_names = ['FPR', 'FNR', 'TPR', 'TNR', 'PPV', 'FPR+FNR', 'J=TPR+TNR-1']

        self.selection = {}
        self.train_folds = {}
        self.train_folds_range = {}
        self.y_labels_train_tab = {}
        self.y_labels_dict = {}
        self.Np_Nn_dict = {}

        # load base data
        mlb = self.load_data(self.data_folder, self.data_name, '', self.eval_params)
        if mlb:
            self.classes = utils.mlb_change_class(mlb).classes_  # class labels list
        else:
            return

        if 'data_set' in self.eval_params:
            eval_params_ds = self.eval_params['data_set']
            #data_folder_ds = os.path.join(norm_output_folder, eval_params_ds['exp_name'], 'data')
            self.data_name_ds = eval_params_ds['data_name']
            mlb = self.load_data(self.data_folder, self.data_name_ds, self.suffix, eval_params_ds)
            if not mlb:
                return

        # load actual class labels
        if not self.load_labels(self.data_name, ''):
            print(self.evaluate_text, '\nError: y_<data_type>.npy file does not exist!')
            self.state = False # False state of object
            return

        # unite train and valid labels (True or False)
        if self.use_train_valid_for_thr:
            self.y_labels_thr = np.concatenate((self.y_labels_dict[self.data_types_ext[0]], self.y_labels_dict[self.data_types_ext[1]]), axis = 0) # 'train' & 'valid'
            self.data_type_name = self.data_types_ext[0] + '_' + self.data_types_ext[1]
            self.data_type_suffix = '_' + self.data_types_ext[0][0] + '_' + self.data_types_ext[1][0]
        else:
            self.y_labels_thr = self.y_labels_dict[self.data_types_ext[0]] # 'train'
            self.data_type_name = self.data_types_ext[0]
            self.data_type_suffix = '_' + self.data_types_ext[0][0]

        self.data_type_name_thr = self.data_type_name + '_thr'
        if 'data_set' in self.eval_params:
            sel_suffix = eval_params['data_set']['sel_suffix']
        self.file_type = 'folds' + self.data_type_suffix + self.suffix + sel_suffix
        self.out_text.append('data_type: ' + self.data_type_name) # data type for pdf first page only

        self.N_labels = self.y_labels_thr.shape[1]
        self.label_inds = range(self.N_labels)
        self.Np_Nn_thr = self.calc_Np_Nn(self.y_labels_thr)

        if 'data_set' in self.eval_params:
            if not self.load_labels(self.data_name_ds, self.suffix):
                print(self.evaluate_text, '\nError: y_<data_type>' + self.suffix + '.npy file does not exist!')
                self.state = False # False state of object
                return

        # Set pd.ExcelWriter object
        self.excel_writer = excel_writer

        # True state of object
        self.state = True

    # return state of object: True of False
    def __bool__(self):
        return self.state

    # load df table, ind matrix and MultiLabelBinarizer object
    def load_data(self, data_folder, data_name, suffix, eval_params):
        mlb_path = os.path.join(data_folder, 'tab_ind_mlb' + suffix + '.pkl')
        if os.path.isfile(mlb_path): # file exists
            with open(mlb_path, 'rb') as tokenizer:
                df_labels_full, y_labels_full, mlb = pickle.load(tokenizer)

                # if swiched on select samples which classes intersect with base set of classes
                if 'samples_of_classes' in eval_params and eval_params['samples_of_classes']:
                    task = 'all_scp' if self.task == 'all' else self.task
                    selection = df_labels_full[task].apply(lambda x: not set(x).isdisjoint(self.classes))
                    df_labels_full_sel = df_labels_full[selection]
                    self.selection[data_name] = {}
                    for data_type in self.data_types_ext:
                        data_type_ds = data_type + suffix
                        if data_type == self.data_types_ext[0]:
                            self.selection[data_name][data_type] = selection[df_labels_full['strat_fold'] <= self.data_folds[data_type]]
                            self.y_labels_dict[data_type_ds] = y_labels_full[df_labels_full_sel['strat_fold'] <= self.data_folds[data_type]]
                        else:
                            self.selection[data_name][data_type] = selection[df_labels_full['strat_fold'] == self.data_folds[data_type]]
                            self.y_labels_dict[data_type_ds] = y_labels_full[df_labels_full_sel['strat_fold'] == self.data_folds[data_type]]
                    df_labels_full = df_labels_full_sel
                elif suffix:
                    df_labels_full = df_labels_full[df_labels_full.all_scp_len > 0]

                train_fold_max = self.data_folds[self.data_types_ext[0]]
                if data_name not in self.selection and y_labels_full.shape[0] == df_labels_full.shape[0]:
                    self.y_labels_train_tab[data_name] = y_labels_full[df_labels_full['strat_fold'] <= train_fold_max] # for checking correctness

                # form data for multi train folds case
                if eval_params['add_train_folds']:
                    self.train_folds[data_name] = df_labels_full.loc[df_labels_full['strat_fold'] <= train_fold_max, 'strat_fold'].to_numpy()
                    self.train_folds_range[data_name] = [(fold, 'tr_fold_' + str(fold)) for fold in range(1, train_fold_max + 1)]

        else:
            mlb_path = os.path.join(data_folder, 'mlb' + suffix + '.pkl')
            if os.path.isfile(mlb_path): # file exists
                with open(mlb_path, 'rb') as tokenizer:
                    mlb = pickle.load(tokenizer)
            else:
                print(self.evaluate_text, '\nError: mlb.pkl file does not exist!')
                self.state = False # False state of object
                mlb = None

        return mlb

    # Np Nn calculation
    def calc_Np_Nn(self, y_labels):

        Np_Nn = np.zeros((y_labels.shape[1], 2), dtype = int)
        for l in range(Np_Nn.shape[0]): # loop for labels
            Np_Nn[l, 0] = np.count_nonzero(y_labels[:, l]) # Np
            Np_Nn[l, 1] = y_labels.shape[0] - Np_Nn[l, 0]  # Nn

        return Np_Nn

    # load actual class labels
    def load_labels(self, data_name, suffix):

        for data_type in self.data_types_ext:
            name = data_type_to_name(data_type) + suffix
            labels_path = os.path.join(self.data_folder, 'y_' + name + '.npy')
            if os.path.isfile(labels_path): # file exists
                data_type_ds = data_type + suffix
                if data_type_ds not in self.y_labels_dict:
                    self.y_labels_dict[data_type_ds] = np.load(labels_path, allow_pickle = True)
                #if data_name in self.selection:
                #    self.y_labels_dict[data_type_ds] = self.y_labels_dict[data_type_ds][self.selection[data_name][data_type]]
                self.Np_Nn_dict[data_type_ds] = self.calc_Np_Nn(self.y_labels_dict[data_type_ds])
            else:
                return False

        # check corectness of y_labels_train
        if data_name in self.y_labels_train_tab:
            assert np.array_equal(self.y_labels_train_tab[data_name], self.y_labels_dict[self.data_types_ext[0] + suffix]), 'y_labels_train is not equal to data from original table!' # 'train'

        # add labels of train folds
        if data_name in self.train_folds_range:
            for (fold, data_type) in self.train_folds_range[data_name]:
                data_type_ds = data_type + suffix
                self.y_labels_dict[data_type_ds] = self.y_labels_dict['train' + suffix][self.train_folds[data_name] == fold]
                self.Np_Nn_dict[data_type_ds] = self.calc_Np_Nn(self.y_labels_dict[data_type_ds])

        return True

    # create head of table for model results
    def create_table(self, Np_Nn, model_txt):
        df_res_table = pd.DataFrame(np.empty(0, dtype = self.dtypes))
        label_ind_arr = [i for i in self.label_inds]
        gap = ['', '', '']
        df_res_table.loc[:, self.dtype_list[0][0]] = [*['', model_txt, 'Mean'], *label_ind_arr] # label_i
        df_res_table.loc[:, self.dtype_list[1][0]] = [*gap, *self.classes]  # label
        df_res_table.loc[:, self.dtype_list[2][0]] = [*gap, *Np_Nn[:, 0]]   # Np
        df_res_table.loc[:, self.dtype_list[3][0]] = [*gap, *Np_Nn[:, 1]]   # Nn

        return df_res_table

    # create head of result table
    def create_res_table(self, data_type_name):
        # header table text
        gap = ['', '']
        return pd.DataFrame({self.dtype_list[0][0]: [self.out_text[0] + ', ' + self.data_exp_text, 'data_type:'], # label_i
                             self.dtype_list[1][0]: gap,   # label
                             self.dtype_list[2][0]: ['', data_type_name],   # Np
                             self.dtype_list[3][0]: gap})  # Nn

    # calc conf matrix for unique sorted thresholds
    def conf_matrix(self, y_true, y_pred):
        inds = np.argsort(y_pred)
        y_pred_uniq = np.unique(y_pred[inds]) # return_index = True
        #inds = range(len(y_pred_uniq))
        conf_m_out = np.zeros((len(y_pred_uniq), 4), dtype = int) # array by y_th and TP, FP, TN, FN

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

            conf_m_out[j][0] = TP_tmp
            conf_m_out[j][1] = FP_tmp
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
                conf_m_out[j][2] = TN_tmp
                conf_m_out[j][3] = FN_tmp
            # print('j = ', j, 'y_pred[i] = ', y_pred[i], ', i = ', i, 'i_prev = ', i_prev)

            i_prev = i

        return conf_m_out, y_pred_uniq

    # build first page
    def build_first_page(self, pdf, text):
        matplotlib.use('agg') # due to this memory leakage problem was solved

        fig, ax_txt = plt.subplots(figsize = self.figsize)

        # text graph
        ax_txt.axis('off')
        ax_txt.text(0.5, 0.7, text, size = 14, ha = 'center', transform = fig.transFigure)

        pdf.savefig()
        # fig.clf()
        plt.close()

    # build graph
    def build_graph(self, pdf, y_pred, fpr_fnr_etc, text):
        matplotlib.use('agg') # due to this memory leakage problem was solved

        fig, (ax_txt, ax1, ax2) = plt.subplots(nrows = 3, ncols = 1, figsize = self.figsize, gridspec_kw = {'height_ratios': [1, 10, 10]})

        y_up_lim = np.max(y_pred)
        if y_up_lim > 1.0:
            y_up_lim = 1.0

        # main graph
        for i in range(len(self.fpr_fnr_names)):
            ax1.plot(y_pred, fpr_fnr_etc[:, i], label = self.fpr_fnr_names[i], linewidth = 1.0)

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
        ax2.set(xlabel = self.fpr_fnr_names[0], ylabel = self.fpr_fnr_names[2], title = 'ROC')
        ax2.set_xticks(ticks)
        ax2.set_yticks(ticks)
        ax2.grid(True, which = 'major', linestyle = '--')

        plt.tight_layout() # adjust graph positions for tight layout

        # text graph
        ax_txt.axis('off')
        ax_txt.text(0.5, 0.91, text, size = 12, ha = 'center', transform = fig.transFigure)

        pdf.savefig()
        # fig.clf()
        plt.close()

    # tp, fp, tn, fn
    def calc_fpr_fnr_etc(self, y_labels_col, y_preds_col):
        conf_m, y_pred_uniq = self.conf_matrix(y_labels_col, y_preds_col)

        fpr_fnr_etc = np.zeros((len(y_pred_uniq), 7), dtype = float)
        fpr_fnr_etc[:, 0] = conf_m[:, 1] / (conf_m[:, 1] + conf_m[:, 2]) # FPR = fp / (fp + tn)
        fpr_fnr_etc[:, 1] = conf_m[:, 3] / (conf_m[:, 0] + conf_m[:, 3]) # FNR = fn / (fn + tp)
        fpr_fnr_etc[:, 2] = 1.0 - fpr_fnr_etc[:, 1] # TPR = 1 - FNR
        fpr_fnr_etc[:, 3] = 1.0 - fpr_fnr_etc[:, 0] # TNR = 1 - FPR
        fpr_fnr_etc[:, 4] = conf_m[:, 0] / (conf_m[:, 0] + conf_m[:, 1]) # PPV = tp / (tp + fp)
        fpr_fnr_etc[:, 5] = fpr_fnr_etc[:, 0] + fpr_fnr_etc[:, 1] # FPR + FNR
        fpr_fnr_etc[:, 6] = fpr_fnr_etc[:, 2] + fpr_fnr_etc[:, 3] - 1.0 # J = TPR + TNR - 1

        return y_pred_uniq, conf_m, fpr_fnr_etc

    # find optimal threshold
    def find_optimal_threshold(self, fpr_fnr_etc):
        # argmin(|FPR - FNR|)
        ind_min1 = np.argmin(np.abs(fpr_fnr_etc[:, 0] - fpr_fnr_etc[:, 1]))
        # argmin(FPR + FNR)
        ind_min2 = np.argmin(fpr_fnr_etc[:, 5])

        return [ind_min1, ind_min2]

    # form output table
    def form_output_table(self, df_mean_res, data_type):
        index_names = df_mean_res.index.values
        mean_dict = {index_names[0]: df_mean_res.loc[index_names[0]]} # 'ROCAUC'
        for i in range(1, len(index_names)):
            name = index_names[i]
            if name[:3] == 'thr':
                continue
            mean_dict[name] = df_mean_res.loc[name]

        return pd.DataFrame(mean_dict, index = [data_type])

    # calc roc_auc, fpr, fnr for every labels,
    # build graphs and estimate thresholds
    # self.y_labels_thr - true relations to classes (labels) n x m
    # y_preds_thr - predicted relations to classes n x m
    # rpath - model path for txt file
    def challenge_metrics_thr(self, y_preds_thr, rpath, model_txt):

        # create df_res_table
        df_res_table = self.create_table(self.Np_Nn_thr, model_txt)

        path_conf_m_table = rpath + self.data_type_name_thr + self.suffix + '_conf_mat'

        if self.eval_params['save_pdf_files']:
            # open pdf file
            pdf = PdfPages(path_conf_m_table + '.pdf')
            # build first page
            first_page_txt = ''
            for i, text in enumerate(self.out_text):
                if i > 0:
                    first_page_txt += '\n'
                first_page_txt += text
            first_page_txt += '\n' + 'model: ' + model_txt
            self.build_first_page(pdf, first_page_txt)

        df_fpr_fnr_table_list = [] # list of tables by labels
        # cycle by labels (columns)
        for l in self.label_inds: # [0, 1, 2]
            y_labels_col = self.y_labels_thr[:, l]
            y_preds_col = y_preds_thr[:, l]

            # tp, fp, tn, fn
            y_pred_uniq, conf_m, fpr_fnr_etc = self.calc_fpr_fnr_etc(y_labels_col, y_preds_col)

            # ROC AUC
            if self.Np_Nn_thr[l, 0] > 0 and self.Np_Nn_thr[l, 1] > 0:
                roc_auc = metrics.roc_auc_score(y_labels_col, y_preds_col)
            else:
                roc_auc = np.nan

            # write to dataframe
            df_fpr_fnr_table = pd.DataFrame(conf_m, columns = ['tp', 'fp', 'tn', 'fn'])
            df_fpr_fnr_table.insert(0, 'threshold', y_pred_uniq)

            # write FPR, FNR, etc to dataframe
            for i in range(len(self.fpr_fnr_names)):
                df_fpr_fnr_table[self.fpr_fnr_names[i]] = fpr_fnr_etc[:, i]

            # find optimal threshold
            ind_min_list = self.find_optimal_threshold(fpr_fnr_etc)

            # write text to pdf
            #Np = np.count_nonzero(y_labels_col)
            #Nn = self.y_labels_thr.shape[0] - Np

            values = [roc_auc]

            for k in range(self.N_thrs): # loop for N_thr
                values.append(y_pred_uniq[ind_min_list[k]])
                values.append(fpr_fnr_etc[ind_min_list[k], 0])
                values.append(fpr_fnr_etc[ind_min_list[k], 1])

            if self.eval_params['save_pdf_files']:
                values_txt = [str(self.Np_Nn_thr[l, 0]),
                              str(self.Np_Nn_thr[l, 1])]
                for val in values:
                    if isinstance(val, float):
                        values_txt.append('{:.4f}'.format(val)) # float
                    else:
                        values_txt.append(str(val)) # str, int and others

                crit_names = ['min(|FPR - FNR|)', 'min(FPR + FNR)']

                text = ('labels[' + str(l) + '] = ' + self.classes[l]
                        + '\nNp = ' + values_txt[0] + ', Nn = ' + values_txt[1]
                        + '\nROC AUC = ' + values_txt[2])
                i = 5
                for k in range(self.N_thrs): # loop for N_thr
                    text += ('\n' + crit_names[k] + ': ' + self.dtype_list[i][0] + ' = ' + values_txt[i - 2]
                    + ', ' + self.dtype_list[i + 1][0] + ' = ' + values_txt[i - 1]
                    + ', ' + self.dtype_list[i + 2][0] + ' = ' + values_txt[i])
                    i += 3

                # write graph to pdf
                self.build_graph(pdf, y_pred_uniq, fpr_fnr_etc, text)

            # write to txt file
            if self.eval_params['save_raw_txt']:
                df_fpr_fnr_table_list.append(df_fpr_fnr_table)

            # write values to table
            df_res_table.iloc[l + 3, 4:] = values

        if self.eval_params['save_pdf_files']:
            # close pdf file
            pdf.close()

        if self.eval_params['save_raw_txt']:
            df_conf_m_res = pd.concat(df_fpr_fnr_table_list, keys = self.label_inds, names = ['label', 'i'])
            df_conf_m_res.to_csv(path_conf_m_table + '.csv', ';', decimal = ',')

        # calc mean values
        df_mean_res = df_res_table.mean(axis = 0, numeric_only = True)
        df_res_table.iloc[2, 4:] = df_mean_res

        return df_res_table, self.form_output_table(df_mean_res, self.data_type_name_thr)

    # calc roc_auc, fpr, fnr
    # y_labels - true relations to classes (labels) n x m
    # y_preds - predicted relations to classes n x m
    # thr_arr - thresholds for estimation k x m
    def challenge_metrics(self, y_labels, y_preds, thr_arr, model_txt, data_type):

        Np_Nn = self.Np_Nn_dict[data_type]
        # create df_res_table
        df_res_table = self.create_table(Np_Nn, model_txt)

        #ROC_AUC
        roc_auc = np.zeros((self.N_labels, 1), dtype = float)
        for l in self.label_inds: # loop for labels
            if Np_Nn[l, 0] > 0 and Np_Nn[l, 1] > 0:
                roc_auc[l, 0] = metrics.roc_auc_score(y_labels[:, l], y_preds[:, l]) # ROC AUC
            else:
                roc_auc[l, 0] = np.nan

        # write values to table
        #df_res_table.iloc[3:, 2:4] = Np_Nn # Np and Nn
        df_res_table.iloc[3:, 4] = roc_auc # ROC AUC

        # FPR, FNR
        column_i = 5
        y_preds_bin = np.zeros(y_preds.shape, dtype = int)
        for k in range(self.N_thrs): # loop for N_thr
            # get binarized preds
            for l in self.label_inds: # loop for labels
                y_preds_bin[:, l] = np.where(y_preds[:, l] >= thr_arr[k, l], 1, 0)

            # conf matices
            conf_matr_thr = metrics.multilabel_confusion_matrix(y_labels, y_preds_bin)

            # FPR FNR
            fpr_fnr_arr = np.zeros((self.N_labels, 2), dtype = float)
            for l in self.label_inds: # loop for labels
                fpr_fnr_arr[l, 0] = conf_matr_thr[l, 0, 1] / Np_Nn[l, 1]  # FPR
                fpr_fnr_arr[l, 1] = conf_matr_thr[l, 1, 0] / Np_Nn[l, 0]  # FNR

            # write values to table
            df_res_table.iloc[3:, column_i] = thr_arr[k, :] # thri
            column_i += 1
            df_res_table.iloc[3:, column_i:(column_i + 2)] = fpr_fnr_arr # FPRi FNRi
            column_i += 2

        # calc mean values
        df_mean_res = df_res_table.mean(axis = 0, numeric_only = True)
        df_res_table.iloc[2, 4:] = df_mean_res

        return df_res_table, self.form_output_table(df_mean_res, data_type)

    # load predictions
    def load_preds(self, mpath, data_name, suffix):
        bypass_model = False
        for data_type in self.data_types_ext:
            name = data_type_to_name(data_type)
            preds_path = mpath + 'y_' + name + '_pred' + suffix + '.npy'
            if os.path.isfile(os.path.normpath(preds_path)): # file exists
                data_type_ds = data_type + suffix
                self.y_preds_dict[data_type_ds] = np.load(preds_path, allow_pickle = True)
                if data_name in self.selection:
                    self.y_preds_dict[data_type_ds] = self.y_preds_dict[data_type_ds][self.selection[data_name][data_type]]
            else: # error
                return False

        return True

    # calc challenge metrics for models
    def challenge_metrics_models(self):

        # initialize lists of result tables
        if self.use_train_valid_for_thr:
            data_types_estim = self.data_types_ext[2:] # 'test'
        else:
            data_types_estim = self.data_types_ext[1:] # 'valid' & 'test'

        #data_types_estim = self.data_types_ext # DEBUG
        # add data types to data_types_estim
        if self.data_name in self.train_folds_range:
            for (_, data_type) in self.train_folds_range[self.data_name]:
                data_types_estim.append(data_type)

        # add validation fold
        if self.use_train_valid_for_thr:
            data_types_estim.append(self.data_types_ext[1])

        if 'data_set' in self.eval_params:
            data_types_ds = [data_type + self.suffix for data_type in self.data_types_ext]
            data_types_estim += data_types_ds

            if self.data_name_ds in self.train_folds_range:
                for (_, data_type) in self.train_folds_range[self.data_name_ds]:
                    data_types_estim.append(data_type + self.suffix)

        #res_table_list = {}
        #for data_type in data_types_estim:
        #    res_table_list[data_type] = [self.create_res_table(data_type)]

        dfs_models = [] # dataframes of models related to exp table

        models = sorted(os.listdir(os.path.join(self.output_folder, self.experiment_name, 'models')), key = str.lower)
        # move 'ensemble' and 'naive' to the end of list
        models.sort(key = lambda s: s == 'ensemble' or s == 'naive')

        # iterate over all models fitted so far
        for model in models:
            #if model != 'fastai_xresnet1d101': # and model != 'fastai_resnet1d_wang'
            #    continue

            model_txt = 'model: ' + model
            caption_text = self.evaluate_text + ', ' + model_txt
            print(caption_text)
            mpath = self.output_folder + self.experiment_name + '/models/' + model + '/'
            rpath = self.output_folder + self.experiment_name + '/models/' + model + '/results/'

            # load predictions
            self.y_preds_dict = {}
            bypass_model = not self.load_preds(mpath, self.data_name, '')

            # bypass model because of loading data error
            if bypass_model:
                continue

            # unite train and valid predictions (True or False)
            if self.use_train_valid_for_thr:
                y_preds_thr = np.concatenate((self.y_preds_dict[self.data_types_ext[0]], self.y_preds_dict[self.data_types_ext[1]]), axis = 0) # 'train' & 'valid'
            else:
                y_preds_thr = self.y_preds_dict[self.data_types_ext[0]] # 'train'

            # calc roc_auc, fpr, fnr and build graphs and thresholds estimation
            df_res_table_thr, df_mean_res_thr = self.challenge_metrics_thr(y_preds_thr, rpath, model)
            #res_table_list_thr.append(df_res_table_thr)
            #df_mean_res_thr.to_csv(rpath + self.data_type_name_thr + '_results' + '.csv')

            # get threshold values
            thr_arr = np.zeros((self.N_thrs, self.N_labels), dtype = float)
            for k in range(self.N_thrs): # loop for N_thr
                str_num = str(k + 1)
                thr_arr[k, :] = df_res_table_thr.loc[3:, 'thr' + str_num].to_numpy()

            if not dfs_models:
                df_head_thr = self.create_res_table(self.data_type_name_thr)
                df_res_table_thr = pd.concat([df_head_thr, df_res_table_thr], ignore_index = True) # dataframes of data types related to model

            dfs_types = [df_res_table_thr] # dataframes of data types related to model
            mean_res_list = [df_mean_res_thr]

            # add predictions of train folds
            if self.data_name in self.train_folds_range:
                for (fold, data_type) in self.train_folds_range[self.data_name]:
                    self.y_preds_dict[data_type] = self.y_preds_dict['train'][self.train_folds[self.data_name] == fold]

            # add predictions of data_set train, valid & test folds
            if 'data_set' in self.eval_params:
                self.load_preds(mpath, self.data_name_ds, self.suffix)

                # add predictions of train folds
                if self.data_name_ds in self.train_folds_range:
                    for (fold, data_type) in self.train_folds_range[self.data_name_ds]:
                        self.y_preds_dict[data_type + self.suffix] = self.y_preds_dict['train' + self.suffix][self.train_folds[self.data_name_ds] == fold]

            # calc estimation for test or valid and test
            for data_type in data_types_estim:
                df_res_table, df_mean_res = self.challenge_metrics(self.y_labels_dict[data_type], self.y_preds_dict[data_type], thr_arr, model, data_type)
                #res_table_list[data_type].append(df_res_table)
                #df_mean_res.to_csv(os.path.join(rpath, data_type + self.data_type_suffix + '_results' + '.csv'))

                if not dfs_models:
                    df_head = self.create_res_table(data_type)
                    df_res_table = pd.concat([df_head, df_res_table], ignore_index = True)

                # remove columns: label_i, label, thr1, thr2
                df_res_table.drop(df_res_table.columns[0:2], axis = 'columns', inplace = True)
                df_res_table.drop(df_res_table.columns[df_res_table.columns.str[0:3] == 'thr'], axis = 'columns', inplace = True)

                dfs_types.append(df_res_table)
                mean_res_list.append(df_mean_res)

            df_cc_types = pd.concat(dfs_types, axis = 'columns')
            dfs_models.append(df_cc_types)

            df_cc_mean_res = pd.concat(mean_res_list)
            df_cc_mean_res.to_csv(os.path.join(rpath, self.file_type + '_results' + '.csv'))

        # form tables and save to txt
        #df_cc_res_table_thr = pd.concat(res_table_list_thr, ignore_index = True)
        #df_cc_res_table_thr.to_csv(os.path.join(self.results_folder, self.data_type_name_thr + '_' + self.experiment_name.lower() + '_results' + '.csv'), ';', index = False, decimal = ',')
        #df_cc_list = [df_cc_res_table_thr]

        #for data_type in data_types_estim:
        #    df_cc_res_table = pd.concat(res_table_list[data_type], ignore_index = True)
        #    #df_cc_res_table.to_csv(os.path.join(self.results_folder, data_type + self.data_type_suffix + '_' + self.experiment_name.lower() + '_results' + '.csv'), ';', index = False, decimal = ',')
        #    df_cc_res_table.drop(df_cc_res_table.columns[0:2], axis = 'columns', inplace = True)
        #    df_cc_res_table.drop(df_cc_res_table.columns[df_cc_res_table.columns.str[0:3] == 'thr'], axis = 'columns', inplace = True)
        #    df_cc_list.append(df_cc_res_table)

        if dfs_models: # dfs_models is not empty
            df_cc_models = pd.concat(dfs_models, ignore_index = True)

            if self.eval_params['save_to_csv']:
                d_types_str = 'results_' + self.file_type + '_' + self.experiment_name.lower() + '_models'
                df_cc_models.to_csv(os.path.join(self.results_folder, d_types_str + '.csv'), ';', index = False, decimal = ',')

            if self.eval_params['save_to_excel'] and self.excel_writer:
                df_cc_models.to_excel(self.excel_writer, sheet_name = self.experiment_name, index = False)
