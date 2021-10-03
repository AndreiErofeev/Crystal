import xgboost as xgb
import pandas as pd
import numpy as np
import pickle

from .featuretransforming import floor_cleaning, feature_transformer

def predict(data_test, modelname = 'model.pkl', outname = '../submit.csv'):

    data_test = floor_cleaning(data_test)
    features = ['id', 'floor', 'city', 'lat', 'lng', 'reform_count_of_houses_1000', 'reform_count_of_houses_500',
                'reform_house_population_1000', 'reform_mean_year_building_1000', 'price_type'] + \
               data_test.filter(regex='osm.*').columns.to_list() + ['total_square', 'realty_type', 'region']
    features.remove('osm_city_nearest_name')

    f_transf_t = feature_transformer(data_test, features, mode='test')
    X_subm = f_transf_t.transform(
        fill_features_mean=['reform_house_population_1000', 'reform_mean_year_building_1000', 'floor'],
        drop_cols=['city', 'id'], cat_features=['realty_type'])
    X_subm = X_subm.drop(columns=['region', 'price_type'])

    with open(modelname, 'rb') as f:
        models = pickle.load(f)

    X_subm['zero_preds'] = models['pt_zeros_model'].predict(X_subm)

    y_subm = np.exp(models['pt_ones_model'].predict(X_subm)) * models['magic_factor']
    df_to_submit = pd.DataFrame(data=np.array([data_test['id'].values, y_subm]).T,
                                columns=['id', 'per_square_meter_price'])
    df_to_submit.to_csv(outname)