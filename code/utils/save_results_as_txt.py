import numpy as np
import os

outputs = [
    'output_develop',
    'output_my',
    'output_drozd'
    ]
exps = [
    'exp0',
    'exp1',
    'exp_ICBEB'
    ]

#path = r'D:\Мои документы\Работа Проекты\Python\ECG_PTBXL\output_develop\exp0\models\fastai_xresnet1d101'
basepath = r'D:\Мои документы\Работа Проекты\Python\ECG_PTBXL' #\output_my\exp1\data
#filename = 'y_test_pred.npy'
filename = 'y_test.npy'

for output in outputs:
    for exp in exps:
        path = os.path.join(basepath, output, exp, 'data')
        y_test = np.load(os.path.join(path, filename), allow_pickle=True)
        np.savetxt(os.path.join(path, filename + '.txt'), y_test)
