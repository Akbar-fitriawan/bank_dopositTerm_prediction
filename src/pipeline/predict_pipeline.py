import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'

            # model
            model = load_object(file_path=model_path)
            #preproc
            preprocessor = load_object(file_path=preprocessor_path)
            data_preproc = preprocessor.transform(features)
            preds = model.predict(data_preproc)
            # Mapping 0 and 1 to 'no' and 'yes'
            prediction = ['yes' if pred == 1 else 'no' for pred in preds]

            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 job: str, 
                 education: str, 
                 contact: str, 
                 poutcome: str, 
                 age_group: str, 
                 duration: float, 
                 campaign: int, 
                 pdays: int, 
                 previous: int,
                 emp_var_rate: float,
                 cons_price_idx: float, 
                 cons_conf_idx: float):
        
    # Menangani input kosong atau tidak valid dengan nilai default
        self.job = job if job != '' else 'unknown'
        self.education = education if education != '' else 'unknown'
        self.contact = contact if contact != '' else 'cellular'
        self.poutcome = poutcome if poutcome != '' else 'nonexistent'
        self.age_group = age_group if age_group != '' else 'mid-career(31-45y)'
        self.duration = float(duration) if duration != '' else 0.0
        self.campaign = int(campaign) if campaign != '' else 1
        self.pdays = int(pdays) if pdays != '' else 999  # Default value for 'pdays'
        self.previous = int(previous) if previous != '' else 0
        self.emp_var_rate = float(emp_var_rate) if emp_var_rate != '' else 0.0
        self.cons_price_idx = float(cons_price_idx) if cons_price_idx != '' else 93.994
        self.cons_conf_idx = float(cons_conf_idx) if cons_conf_idx != '' else -36.4

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'job': [self.job],
                'education': [self.education],
                'contact': [self.contact],
                'poutcome': [self.poutcome],
                'age_group': [self.age_group],
                'duration': [self.duration],
                'campaign': [self.campaign],
                'pdays': [self.pdays],
                'previous': [self.previous],
                'emp_var_rate': [self.emp_var_rate],
                'cons_price_idx': [self.cons_price_idx],
                'cons_conf_idx': [self.cons_conf_idx]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
    
        