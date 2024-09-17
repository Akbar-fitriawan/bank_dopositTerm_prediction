import os
import sys
import pandas as pd
import numpy as np
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