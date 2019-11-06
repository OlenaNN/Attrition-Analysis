from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from hyperopt import STATUS_OK



def lgb_model(space, train_x, train_y, test_x, test_y,early_stopping_rounds=50):
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

    lgbm.fit(train_x, train_y,
             eval_set=[(train_x, train_y), (test_x, test_y)],
             early_stopping_rounds=early_stopping_rounds,
             eval_metric='auc',
             verbose=False)

    predictions = lgbm.predict(test_x)
    test_preds = lgbm.predict_proba(test_x)[:, 1]
    train_preds = lgbm.predict_proba(train_x)[:, 1]

    train_auc = roc_auc_score(train_y, train_preds)
    test_auc = roc_auc_score(test_y, test_preds)
    accuracy = accuracy_score(test_y, predictions)

    return {'status': STATUS_OK, 'loss': 1 - test_auc, 'accuracy': accuracy,
            'test auc': test_auc, 'train auc': train_auc
            }