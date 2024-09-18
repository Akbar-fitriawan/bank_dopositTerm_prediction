# import os
# import sys
# from dataclasses import dataclass

# # Pemodelan ML
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier

# # Evaluasi model
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
# # ----------------------------------------------#
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold

# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_object

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path=os.path.join("artifacts", "model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()


#     def initia
