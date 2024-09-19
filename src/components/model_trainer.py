import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib

from imblearn.over_sampling import SMOTE as smote
from imblearn.pipeline import Pipeline as ImbPipeline

# Pemodelan ML
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier



# Evaluasi model
from sklearn.metrics import (accuracy_score, 
                             precision_score,
                             recall_score,
                             f1_score,
                             classification_report,
                             confusion_matrix,
                             roc_curve,
                             roc_auc_score,
                             precision_recall_curve,
                             average_precision_score)
# ----------------------------------------------#
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


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

            # get pipeline models
            seed = 42
            np.random.seed(seed)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            trained_models = {}
            cv_results = []

            for model_name, model in models.items():
            
                pipe_model = ImbPipeline(steps=[
                    # ('preproc', preprocessor),
                    ('oversample', smote(random_state=seed)),
                    ('model', model)
                ])

                # Training the model
                pipe_model.fit(X_train, y_train)

                # cross validataion
                cv_scores = cross_val_score(pipe_model, X_train, y_train, cv=skf, scoring='accuracy')

                # predict to evaluate
                y_pred = pipe_model.predict(X_test)

                # Menyimpan model yang telah dilatih
                trained_models[model_name] = pipe_model

                cv_results.append({
                'model_name': model_name,
                'avg_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                })

                print(model_name)
                print(f'Accuracy Score : {accuracy_score(y_test, y_pred):.4f}')
                print(f"CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                print(cv_scores)
                print('-' * 30, '\n')
            
            # get best model without cross val
            cv_results_df = pd.DataFrame(cv_results)

            best_model_row = cv_results_df.loc[cv_results_df['avg_score'].idxmax()]
            best_model_name = best_model_row['model_name']
            best_model_score = best_model_row['avg_score']

            best_model = trained_models[best_model_name]

            if best_model_score < 0.7:
                raise CustomException("No best model found")
            logging.info(f"Best model found with high cross validation score")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            prediction = best_model.predict(X_test)

            acc = accuracy_score(y_test, prediction)
            

            return acc

            # predicted = best_model.predict(X_test)
            # acc = accuracy_score(y_test, predicted)
        except Exception as e:
            raise CustomException(e, sys)
