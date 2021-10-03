from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle

from .featuretransforming import floor_cleaning, feature_transformer
from .xgboosttuning import xgb_parametres_selection
from .deviationmetric import deviation_metric

def fine_tinung(zero_regr, ones_regr, X_test, y_ones = None):
    fin_coef = 1.0
    best_score = 100

    preds_for_0_test = zero_regr.predict(X_test)
    X_test_ext = X_test.copy()
    X_test_ext['zero_preds'] = preds_for_0_test

    for balance_coeff in np.arange(0.9, 1.01, 0.01):
        score = deviation_metric(np.array(np.exp(y_ones)),
                                 balance_coeff * np.array(np.exp(ones_regr.predict(X_test_ext))))
        if score < best_score:
            fin_coef = balance_coeff
    return fin_coef


def train(data_train, test_inds, grid_zeros, grid_ones, savepath = 'model.pkl', SEED = 69):
    data_train = floor_cleaning(data_train)
    df_test_0 = data_train[data_train['id'].isin(test_inds['pt_0'])]
    df_test_1 = data_train[data_train['id'].isin(test_inds['pt_1'])]

    df_train = data_train[~data_train['id'].isin(test_inds['pt_1'])]
    df_train = df_train[~df_train['id'].isin(test_inds['pt_0'])]

    features = ['id', 'floor', 'city', 'lat', 'lng', 'reform_count_of_houses_1000', 'reform_count_of_houses_500',
                'reform_house_population_1000', 'reform_mean_year_building_1000', 'price_type'] +\
    df_train.filter(regex = 'osm.*').columns.to_list() + ['total_square', 'realty_type', 'region', 'per_square_meter_price']

    features.remove('osm_city_nearest_name')

    F_trans_train = feature_transformer(df_train, features)
    df_train = F_trans_train.transform(
        fill_features_mean=['floor', 'reform_house_population_1000', 'reform_mean_year_building_1000'],
        cat_features=['realty_type'])

    df_train = df_train.drop(columns=['city', 'id'])

    F_trans_test_0 = feature_transformer(df_test_0, features)
    df_test_0 = F_trans_test_0.transform(
        fill_features_mean=['floor', 'reform_house_population_1000', 'reform_mean_year_building_1000'],
        drop_cols=['city', 'id'], cat_features=['realty_type'])

    F_trans_test_1 = feature_transformer(df_test_1, features, mode='test')
    df_test_1 = F_trans_test_1.transform(
        fill_features_mean=['floor', 'reform_house_population_1000', 'reform_mean_year_building_1000'],
        drop_cols=['city', 'id'], cat_features=['realty_type'])

    df_train = df_train.dropna()
    df_train, df_val = train_test_split(df_train, test_size=0.2, stratify=df_train['region'], random_state=SEED)

    X_train, y_train = df_train.drop(columns=['region', 'per_square_meter_price']), np.log(
        df_train['per_square_meter_price'])
    X_val, y_val = df_val.drop(columns=['region', 'per_square_meter_price']), np.log(df_val['per_square_meter_price'])
    X_test_0, y_test_0 = df_test_0.drop(columns=['region', 'per_square_meter_price']), np.log(
        df_test_0['per_square_meter_price'])
    X_test_1, y_test_1 = df_test_1.drop(columns=['region', 'per_square_meter_price']), np.log(
        df_test_1['per_square_meter_price'])

    train_pt_0 = X_train[X_train['price_type'] == 0].index
    val_pt_0 = X_val[X_val['price_type'] == 0].index

    X_train = X_train.loc[train_pt_0].drop(columns=['price_type'])
    X_val = X_val.loc[val_pt_0].drop(columns=['price_type'])
    X_test_0 = X_test_0.drop(columns=['price_type'])
    X_test_1 = X_test_1.drop(columns=['price_type'])

    best_set_zero_type = xgb_parametres_selection(X_train, X_val, y_train.loc[train_pt_0], y_val.loc[val_pt_0],
                                                  grid_zeros, np.ones(X_train.shape[0]), np.ones(X_val.shape[0]),
                                                  lr=0.1, n_splits=4, n_est=12000)

    best_zero_regr = xgb.XGBRegressor(**best_set_zero_type)
    best_zero_regr.fit(pd.concat([X_train, X_val]), pd.concat([y_train.loc[train_pt_0], y_val.loc[val_pt_0]]),
                       sample_weight=np.ones(pd.concat([X_train, X_val]).shape[0]))

    y_pred_test_0 = best_zero_regr.predict(X_test_0)
    y_pred_test_1 = best_zero_regr.predict(X_test_1)

    train_pt_1 = df_train[df_train['price_type'] == 1].index
    val_pt_1 = df_val[df_val['price_type'] == 1].index
    preds_for_0 = best_zero_regr.predict(df_train.drop(columns=['region', 'per_square_meter_price',
                                                                'price_type']).loc[train_pt_1])
    preds_for_0_val = best_zero_regr.predict(df_val.drop(columns=['region', 'per_square_meter_price',
                                                                  'price_type']).loc[val_pt_1])

    X_ones_train = df_train.loc[train_pt_1].drop(columns=['region', 'per_square_meter_price', 'price_type'])
    X_ones_train['zero_preds'] = preds_for_0

    X_ones_val = df_val.loc[val_pt_1].drop(columns=['region', 'per_square_meter_price', 'price_type'])
    X_ones_val['zero_preds'] = preds_for_0_val

    best_set_ones_type = xgb_parametres_selection(X_ones_train, X_ones_val, y_train.loc[train_pt_1],
                                                  y_val.loc[val_pt_1], grid_ones, np.ones(X_ones_train.shape[0]),
                                                  np.ones(X_ones_val.shape[0]), lr=0.01, n_splits=4, n_est=12000)
    best_ones_regr = xgb.XGBRegressor(**best_set_ones_type)

    best_ones_regr.fit(pd.concat([X_ones_train, X_ones_val]), pd.concat([y_train.loc[train_pt_1], y_val.loc[val_pt_1]]),
                       sample_weight=np.ones(pd.concat([X_ones_train, X_ones_val]).shape[0]))

    magic_prefactor = fine_tinung(best_zero_regr, best_ones_regr, X_test_1, y_pred_test_1)

    solution_dict = {
        'pt_zeros_model':best_zero_regr,
        'pt_ones_model':best_ones_regr,
        'magic_factor':magic_prefactor
    }

    with open(savepath, 'wb') as f:
        pickle.dump(solution_dict, f)