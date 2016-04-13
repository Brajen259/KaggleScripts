import pandas as pd
import sklearn
import scipy
import numpy
from unbalanced_dataset.over_sampling import SMOTE

train = pd.DataFrame.from_csv('train.csv')
features = [ 'var38',
'var15',
'saldo_var30',
'saldo_medio_var5_hace2',
'saldo_medio_var5_hace3',
'num_var22_ult1',
'num_var22_ult3',
'num_var45_hace3',
'saldo_medio_var5_ult3',
'num_var22_hace3']
X = train[features]
Y = train['TARGET']

sm = SMOTE(kind='regular', verbose='verbose')
svmx, svmy = sm.fit_transform(X, Y)
print len(svmx)
