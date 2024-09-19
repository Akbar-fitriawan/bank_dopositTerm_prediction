import sys
import os

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# libraries Data Preprocessing
from sklearn.preprocessing import StandardScaler # scalling
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    prepocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        fungsi untuk transformasi data
        """
        try:
            num_feature = ['duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx']
            one_hot_feature = ['contact', 'poutcome']
            ord_feature = ['age_group','job', 'education']

            ord_transformer = Pipeline(steps=[
               ("ord_encode", OrdinalEncoder())
            ])

            one_hot_transformer = Pipeline(steps=[
               ("onehot_encode", OneHotEncoder())
            ])

            num_transformer = Pipeline(steps=[
               ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_transformer, num_feature),
                    ('onehot', one_hot_transformer, one_hot_feature),
                    ('ordinal', ord_transformer, ord_feature)])

            logging.info("Apply tranformation data")
           
            return preprocessor 
        except Exception as e:
            raise CustomException(e, sys)
            

    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            # read data dari direktori
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preproc=self.get_data_transformer_object()

            target = "y"

            # train df
            input_train_df = train_df.drop(columns=[target], axis=1)
            target_train_series = train_df[target].map({'no':0, 'yes':1})

            # test df
            input_test_df = test_df.drop(columns=[target], axis=1)
            target_test_series = test_df[target].map({'no':0, 'yes':1})

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            # preprocessing apply
            input_feature_train_arr=preproc.fit_transform(input_train_df)
            input_feature_test_arr=preproc.transform(input_test_df)


            train_arr = np.c_[input_feature_train_arr, target_train_series]
            test_arr = np.c_[input_feature_test_arr, target_test_series]

            save_object(
                file_path = self.data_transformation_config.prepocessor_obj_file_path,
                obj = preproc
            )

            return (train_arr, test_arr, self.data_transformation_config.prepocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)


