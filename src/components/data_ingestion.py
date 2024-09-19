import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.clean import clean_data, outlier_handler, imputer_missing_value

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")

        try:
            # read data 
            df = pd.read_csv("notebook\data\\bank_marketing.csv", sep=';', na_values='unknown')
            logging.info("Read the data as dataframe")

            # cleaning data
            df = clean_data(data=df)
            logging.info("Cleaning data and selected feature import")


            # buat folder data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # inisialisasi data raw
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # split data
            logging.info("Train test split initiate")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # handling missing value
            num_cols = ['duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx']
            cat_cols = ['age_group','job', 'education', 'contact', 'poutcome']

            train_set, test_set = imputer_missing_value(train_set, test_set, num_cols, cat_cols)

            # handling outlier
            outlier = ['duration', 'campaign', 'cons_conf_idx']
            train_set, test_set = outlier_handler(train_set, test_set, cols_outlier=outlier)

            # export ke folder data train
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # export ke folder data test
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("ingestion of the data is complate")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
