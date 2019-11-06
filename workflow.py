import pandas as pd
import numpy as np

from hyperopt import hp, tpe, Trials
from hyperopt import fmin

import warnings

import models
import model_report
import data_processing
import data_io





warnings.filterwarnings('ignore')
attrition = pd.read_csv('input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
numerical_data , categorical_data, data_description = data_processing.generate_data_description(attrition)
data_processed = data_processing.data_preparation(attrition, categorical_data)

def train_model(data):
    x_train, x_test, y_train, y_test = data_processing.split_labeled_data_on_train_test(data)

    trials = Trials()
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
        'n_estimators': hp.quniform('n_estimators', 50, 1200, 25),
        'max_depth': hp.quniform('max_depth', 1, 15, 1),
        'num_leaves': hp.quniform('num_leaves', 10, 150, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'feature_fraction': hp.uniform('feature_fraction', .3, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'min_split_gain': hp.uniform('min_split_gain', 0.0001, 0.1)
    }
    lgb_objective = models.lgb_model(x_train, x_test, y_train, y_test,early_stopping_rounds=50)

    lgb_hyperparams = fmin(fn=lgb_objective,
                           max_evals=150,
                           trials=trials,
                           algo=tpe.suggest,
                           space=space
                           )


    lgb_results = model_report.generate_model_report (trials.trials, lgb_hyperparams, 'LightGBM')
    return lgb_results, x_train, x_test

final_model, final_test_data, final_train_data = train_model(data_processed)

model_artifacts = data_io.ModelArtifactsIO

model_version = model_artifacts.build_models_version()
data_saver = model_artifacts.build_test_dataset_path(model_version)


