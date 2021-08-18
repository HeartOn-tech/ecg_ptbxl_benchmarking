import os
import pandas as pd
path = r'D:\Temp\ECG\ICBEB'
table = pd.read_csv(os.path.join(path, 'icbeb_database.csv'))

label_dict = {1:'NORM', 2:'AFIB', 3:'1AVB', 4:'CLBBB', 5:'CRBBB', 6:'PAC', 7:'VPC', 8:'STD_', 9:'STE_'}
result = {}

for ind, scp in label_dict.items(): # цикл по меткам класса
    result[scp] = [0] * 10

for i in range(len(table)): # цикл по строкам таблицы
    scp = list(eval(table.loc[i, 'scp_codes']).keys())[0]
    fold = int(table.loc[i, 'strat_fold'])
    result[scp][fold - 1] += 1

for label, folds in result.items(): # цикл по меткам класса
    print('label:', label, ', folds: ', folds)
