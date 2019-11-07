import os
from hyperopt import hp, tpe, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
import numpy as np


class LGBMModel:
    def __init__(self, load_path=None):
        self.space = {
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
        self.trials = Trials()
        self.model_name = 'LightGBM'
        self.trials_ = self.trials.trials

        if load_path:
            self.model = self.load(load_path)

    @staticmethod
    def load(dir_path):
        """
        :param dir_path: see __init__
        :return: encoder and model
        """
        # with open(os.path.join(dir_path, 'architecture.json'), 'r') as file_:
        #     model = keras.models.model_from_json(json.load(file_))
        # model.load_weights(os.path.join(dir_path, 'model_weights.h5'))
        model = 1
        return  model

    def split_data(self, data):
        target_data = data['Attrition'].reset_index(drop=True)
        model_data = data.drop(columns='Attrition')
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(model_data, target_data, test_size=0.20)
        return self.train_x, self.test_x, self.train_y, self.test_y

    def lgb_objective(self, space, early_stopping_rounds=50):
        lgbm = LGBMClassifier(
            learning_rate=space['learning_rate'],
            n_estimators=int(space['n_estimators']),
            max_depth=int(space['max_depth']),
            num_leaves=int(space['num_leaves']),
            colsample_bytree=space['colsample_bytree'],
            feature_fraction=space['feature_fraction'],
            reg_lambda=space['reg_lambda'],
            reg_alpha=space['reg_alpha'],
            min_split_gain=space['min_split_gain']
            )

        lgbm.fit(self.train_x, self.train_y,
                 eval_set=[(self.train_x, self.train_y), (self.test_x, self.test_y )],
                 early_stopping_rounds=early_stopping_rounds,
                 eval_metric='auc',
                 verbose=False)

        predictions = lgbm.predict(self.test_x)
        test_preds = lgbm.predict_proba(self.test_x)[:, 1]
        train_preds = lgbm.predict_proba(self.train_x)[:, 1]

        train_auc = roc_auc_score(self.train_y, train_preds)
        test_auc = roc_auc_score(self.test_y , test_preds)
        accuracy = accuracy_score(self.test_y , predictions)

        return {'status': STATUS_OK, 'loss': 1 - test_auc, 'accuracy': accuracy,
                'test auc': test_auc, 'train auc': train_auc
                }

    def generate_model_report(self, hyperparams):
        fit_idx = -1
        for idx, fit in enumerate(self.trials_):
            hyp = fit['misc']['vals']
            xgb_hyp = {key: [val] for key, val in hyperparams.items()}
            if hyp == xgb_hyp:
                fit_idx = idx
                break

        train_time = str(self.trials_[-1]['refresh_time'] - self.trials_[0]['book_time'])
        acc = round(self.trials_[fit_idx]['result']['accuracy'], 3)
        train_auc = round(self.trials_[fit_idx]['result']['train auc'], 3)
        test_auc = round(self.trials_[fit_idx]['result']['test auc'], 3)

        results = {
            'model': self.model_name,
            'parameter search time': train_time,
            'accuracy': acc,
            'test auc score': test_auc,
            'training auc score': train_auc,
            'parameters': hyperparams
            }
        return results


    def fit_final_model(self, parameters):
        final_lgbm_model = LGBMClassifier(
            learning_rate=parameters['learning_rate'],
            n_estimators=int(parameters['n_estimators']),
            max_depth=int(parameters['max_depth']),
            num_leaves=int(parameters['num_leaves']),
            colsample_bytree=parameters['colsample_bytree'],
            feature_fraction=parameters['feature_fraction'],
            reg_lambda=parameters['reg_lambda'],
            reg_alpha=parameters['reg_alpha'],
            min_split_gain=parameters['min_split_gain'],
            )
        print('Starting training...')
        final_lgbm_model.fit(self.train_x, self.train_y,
                             eval_set=[(self.train_x, self.train_y,), (self.test_x, self.test_y)],
                             early_stopping_rounds=50,
                             eval_metric='auc',
                             verbose=False)
        return final_lgbm_model