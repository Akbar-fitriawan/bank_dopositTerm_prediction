import sys
import os

import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    prepocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        fungsi untuk transformasi data
        """
        try:
           num_feature = ['duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx']
           cat_feature = ['job','education','contact','poutcome','age_group']

           preprocessor = Pipeline([
               # handling missing value imputer
               ('cat_imputer', CategoricalImputer(imputation_method="frequent", variables=cat_feature)),
               ('num_imputer', MeanMedianImputer(imputation_method="median", variables=num_feature)),

               # handling outlier
               ('winsorizer', Winsorizer(capping_method='iqr', fold=1.5, variables=['duration', 'campaign'])),

               # encoder value
               ('ord_encoder', OrdinalEncoder(encoding_method='arbitrary', variables=['age_group','job', 'education'])), # ordinal encoder note: saya tidak peduli tenteng urutan yang penting komputer membaca polanya :)
               ('one_hot_encoder', OneHotEncoder(variables=['contact', 'poutcome'])), # one hot encoder
               
               # scaling data numerik
               ('scaler', SklearnTransformerWrapper(transformer=StandardScaler(), variables=num_feature))
           ])

           logging.info("Apply tranformation data")
           
           return preprocessor
           
        except Exception as e:
            raise CustomException(e, sys)
            

    def transform_data(self, train_path, test_path):
        
        try:
            # read data dari direktori
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preproc = self.get_data_transformation_object()

            # train df
            input_train_df = train_df.drop(columns='y', axis=1)
            target_train_series = train_df['y']

            # test df
            input_test_df = test_df.drop(columns='y', axis=1)
            target_test_series = test_df['y']

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            # apply preprocessing
            input_feature_train = preproc.fit_transform(input_train_df)
            input_feature_test = preproc.transform(input_test_df)

            train_df_result = pd.concat([input_feature_train, target_train_series], axis=1)
            test_df_result = pd.concat([input_feature_test, target_test_series], axis=1)

            save_object(
                file_path = self.data_transformation_config.prepocessor_obj_file_path,
                obj = preproc
            )

            return (train_df_result, test_df_result, self.data_transformation_config.prepocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)



