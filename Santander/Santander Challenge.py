import pandas as pd
import sklearn
import scipy
import numpy
from unbalanced_dataset.over_sampling import SMOTE


train = pd.DataFrame.from_csv('train.csv')
