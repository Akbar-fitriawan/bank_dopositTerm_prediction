import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OneHotEncoder


from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    prepocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def transformer_pipeline(self):
        """
        fungsi untuk transformasi data
        """
        try:
           num_feature = ['duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx']
           cat_feature = ['job','education','contact','poutcome','age_group', 'y']

           preprocessor = Pipeline([('cat_imputer', CategoricalImputer(imputation_method='frequent', variables=['gender'])),
               
           ])
           

           
        except:
            pass


