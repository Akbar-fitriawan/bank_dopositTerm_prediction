import os
import sys
from dataclasses import dataclass
# import numpy as np
# import pandas as pd


# Pemodelan ML
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
# ----------------------------------------------#

from sklearn.metrics import f1_score, make_scorer, roc_auc_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, train_models, grid_search_tuning_hyperparameter


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            # Define models
            models = {
                'Logistic Regression': LogisticRegression(),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Gaussian Naive Bayes': GaussianNB(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(), # boosting algorithm
                'Xgboost' : XGBClassifier(),
                'SVM': SVC() 
            }

            # traning models
            print("Training models...")
            base_model_tebaik = train_models(models, X_train, y_train, X_test, y_test)

            # Define parameter grids for GridSearchCV 
            param_grids = {
                'Logistic Regression': {
                    'model__C': [0.01, 0.1, 1., 10.],  # Mengurangi jumlah C untuk runtime lebih singkat
                    'model__solver': ['liblinear'],  # Fokus pada solver cepat dan umum untuk dataset kecil-menengah
                    'model__penalty': ['l1', 'l2'],
                    'model__class_weight': ['balanced']
                },
                
                'KNN': {
                    'model__n_neighbors': [5, 7, 9],  # Rentang tetangga lebih kecil untuk efisiensi
                    'model__weights': ['uniform', 'distance'],  # Pertahankan kedua opsi bobot
                    'model__p': [1, 2]  # Manhattan dan Euclidean distance
                },
                
                'Decision Tree': {
                    'model__max_depth': [5, 7, 9],  # Mengurangi kedalaman maksimal
                    'model__min_samples_split': [2, 4],  # Membatasi variasi split
                    'model__min_samples_leaf': [1, 2],  # Mengurangi jumlah sampel daun
                    'model__class_weight': ['balanced'],
                    'model__max_features': ['sqrt', 'log2']  # Menghilangkan opsi 'auto' untuk runtime lebih singkat
                },
                
                'Gaussian Naive Bayes': {
                    'model__var_smoothing': [1e-9, 1e-8, 1e-7]  # Variasi smoothing untuk stabilitas model
                },

                'Random Forest': {
                    'model__n_estimators': [100, 200],  # Kurangi jumlah estimator untuk runtime lebih singkat
                    'model__max_depth': [5, 7],  # Hanya dua opsi kedalaman untuk efisiensi
                    'model__min_samples_split': [2, 4],
                    'model__min_samples_leaf': [1, 2]
                },
                
                'Gradient Boosting': {
                    'model__n_estimators': [100, 200],  # Jumlah estimator yang lebih kecil
                    'model__learning_rate': [0.01, 0.1],  # Fokus pada rentang learning rate yang sering digunakan
                },
                
                'Xgboost': {
                    'model__n_estimators': [100, 200],  # Mengurangi jumlah estimator
                    'model__learning_rate': [0.01, 0.1],  # Rentang learning rate umum untuk runtime efisien
                }, 

                'SVM': {
                    'model__C': [0.1, 1, 10],  # Rentang C yang umum digunakan untuk kontrol regularisasi
                    'model__kernel': ['linear', 'rbf'],  # Dua kernel umum, fokus pada performa yang efisien
                    'model__gamma': ['scale', 'auto'],  # Parameter gamma untuk kernel RBF
                    'model__class_weight': ['balanced']
                }
            }


            # Parameter tuning
            print("Tuning models...")
            scoring = make_scorer(roc_auc_score) # score bisa di ubah sesuai kebutuhan 

            best_model = grid_search_tuning_hyperparameter(
                                                              models,
                                                              X_train,
                                                              y_train,
                                                              param_grids,
                                                              scoring=scoring,)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            

        except Exception as e:
            raise CustomException(e, sys)
