import os
import sys

import numpy as np
import pandas as pd
import joblib
from pprint import pprint as pprint


from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE as smote
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def train_models(models, X_train, y_train, X_test, y_test,):    
    # get pipeline models
    seed = 42
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    trained_models = {}
    cv_results = []

    for model_name, model in models.items():
    
        pipe_model = ImbPipeline(steps=[
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

    prediction = best_model.predict(X_test)
    acc = accuracy_score(y_test, prediction)
    print(f'Best model : {best_model_name}, with accuracy {acc}')
    
    return best_model

# Hyperparameter tuning dengan grid-search
def grid_search_tuning_hyperparameter(models, X, y, param_grids, scoring='accuracy'):
    seed = 42
    np.random.seed(seed)

    tuned_models = {}
    best_params_list = []
    report_list = []
    results_score = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for model_name, model in models.items():
        param_grid = param_grids.get(model_name, {})

        try:
            print(f"Starting Grid Search for {model_name}...")

            model_pipe = ImbPipeline(steps=[
                ('oversample', smote(random_state=seed)),
                ('model', model)
            ])

            grid_search = GridSearchCV(
                estimator=model_pipe,
                param_grid=param_grid,
                cv=skf,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X, y)

            tuned_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            report = grid_search.cv_results_

            print(f"{model_name} - Best Parameters:")
            pprint(best_params)
            print(f"Best Score: {best_score:.4f}")

            tuned_models[model_name] = tuned_model
            best_params_list.append({model_name : best_params})
            results_score.append({'model_name': model_name, 'scores': best_score})
            report_list.append(report)

        except Exception as e:
            logging.error(f"Error Training {model_name}: {e}")

    best_model_row = pd.DataFrame(results_score).loc[pd.DataFrame(results_score)['scores'].idxmax()]
    best_model_name = best_model_row['model_name']
    best_model = tuned_models[best_model_name]

    print(f'Best model : {best_model_name}, with high scores {best_model_row["scores"]}')
    return best_model
