import xgboost as xgb
from sklearn.model_selection import GridSearchCV, ShuffleSplit

def xgb_parametres_selection(X_train, X_val, y_train, y_val, param_grid, train_weights, val_weights,
                             n_splits=10, lr=0.001, n_est=5000):
    regr = xgb.XGBRegressor()
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
    gs = GridSearchCV(regr, param_grid, cv=cv, n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train, sample_weight=train_weights)

    best_param = gs.best_params_

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    dval = xgb.DMatrix(X_val, label=y_val, weight=val_weights)

    best_param['learning_rate'] = lr
    bst = xgb.train(best_param, dtrain, num_boost_round=n_est, early_stopping_rounds=20,
                    evals=[(dval, 'eval')], verbose_eval=False)

    best_param['n_estimators'] = bst.best_iteration
    return best_param