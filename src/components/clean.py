import os
import sys
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer

from src.exception import CustomException

## Buat feature group_age
def age_group(age):
  if age <= 30:
    return 'youth segment(17-30y)'
  elif age > 30 and age <= 45:
    return 'mid-career(31-45y)'
  elif age > 45 and age < 65:
    return 'pre-retirement(46-64y)'
  elif age >= 65:
    return 'retirees(65+)'
  else:
    return 'unknown'

## clean proses
def clean_data(data):
    """
    fungsi untuk clean data 
    """
    try:

        # hapus data duplikat yang terakhir
        data.drop_duplicates(keep='first', inplace=True)

        # Ubah nama kolom
        data.columns = data.columns.str.lower().str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')


        # buat age_group berdasarkan kolom age
        data['age_group'] = data['age'].apply(age_group) # apply fungsi age_group

        # drop age
        data.drop('age', axis=1, inplace=True)

        # Ubah 'duration' ke menit
        data['duration'] = data['duration'] / 60  # Mengubah detik ke menit

        # ubah pdays '999' jadi '0' mengacu kepada customer yang belum di hubungi
        data['pdays'] = data['pdays'].replace(999, 0)

        # ubah data label target jadi 0 dan 1, 
        data['y'].map({'no':0, 'yes':1})

        # drop kolom tidak relevan
        feature_selected = ['job',
                            'education',
                            'contact',
                            'poutcome',
                            'age_group',
                            'duration',
                            'campaign',
                            'pdays',
                            'previous',
                            'emp_var_rate',
                            'cons_price_idx',
                            'cons_conf_idx',
                            'y']
        
        data = data[feature_selected]

    except Exception as e:
        raise CustomException(e, sys)
    
    return data

def imputer_missing_value(X_train, X_test,num_cols=None, cat_cols=None):
    # Mengimputasi kolom numerik
    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    # Mengimputasi kolom kategorikal
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
    
    return X_train, X_test
   



def outlier_handler(X_train, X_test, cols_outlier):
  #cek data normal dan tidak normal
  list_cols_normal = []
  list_cols_Not_normal = []

  for col in X_train[cols_outlier]:

    skew = X_train[col].skew()
    kurtosis = X_train[col].kurt()

    if -0.5 < skew < 0.5 and kurtosis < 3:
      list_cols_normal.append(col)
      
    else:
      list_cols_Not_normal.append(col)

  # handling outlier
  winsorizer_normal_dist = Winsorizer(capping_method='gaussian',
                                      tail='both',
                                      fold=3,
                                      variables=list_cols_normal,
                                      missing_values='ignore')

  winsorizer_not_normal_dist = Winsorizer(capping_method='iqr',
                                 tail='both',
                                 fold=1.5,
                                 variables=list_cols_Not_normal,
                                 missing_values='ignore')

  X_train = winsorizer_normal_dist.fit_transform(X_train)
  X_test = winsorizer_normal_dist.transform(X_test)


  X_train = winsorizer_not_normal_dist.fit_transform(X_train)
  X_test = winsorizer_not_normal_dist.transform(X_test)


  return X_train, X_test
