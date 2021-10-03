from scipy.spatial import cKDTree as KDTree
import pandas as pd
import numpy as np
import tqdm

class feature_transformer():
    def __init__(self, df, features, mode = 'train'):
        self.df = df
        self.features = features
        self.df_out = self.df[features]
        self.mode = mode

    def subway_cut(self, cut_dist:float = 30) -> None:
        '''
        We found the osm_subway_closest_dist is the very important feature and to make model get more information
        from it, researched that only 30 km or less is important for the price. Therefore, higher values just
        replaced with one huge value for easier splittingg of tree-based algorithm.
        :param cut_dist:
        :return:
        '''
        self.df_out.loc[(self.df_out['osm_subway_closest_dist'] > cut_dist), 'osm_subway_closest_dist'] = 10000

    @staticmethod
    def reject_outliers(input, iq_range:float=0.95):
        '''
        Function to skip otliers beyond 2-sigma interval - ONLY FOR TRAINING!
        :param input:
        :param iq_range:
        :return:
        '''
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = input.dropna().quantile([pcnt, 0.50, 1 - pcnt])
        return input[(input < qhigh) & (input > qlow)]

    def clean_outl(self):
        '''
        Outliers removal (5%) is provided separately in each region.
        '''
        for i, city in enumerate(self.df_out['region'].unique()):
            df_city = self.df_out[self.df_out['region'] == city]
            df_city = df_city.loc[self.reject_outliers(df_city['per_square_meter_price']).index]
            if i == 0:
                df_new_out = df_city.copy()
            else:
                df_new_out = pd.concat([df_new_out, df_city])
        self.df_out = df_new_out.reset_index(drop=True)


    def combine_dist_features(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        Function to weight features according to distances and reduce feature space.
        :param df: imput DataFrame
        '''
        list_to_find = ['catering', 'shops', 'offices', 'finance', 'healthcare', 'leisure', 'historic', 'osm_building',
                        'hotels', 'culture', 'amenity', 'train_stop_points', 'transport', 'crossing_points']

        for i in list_to_find:
            list_cur = [col for col in df.columns if i in col]
            print(i, len(list_cur))

            if i == 'leisure':
                df.loc[df[list_cur].iloc[:, 1] == 'S5041', list_cur[1]] = 0
                df.loc[df[list_cur].iloc[:, 2] == '2020-06-07', list_cur[2]] = 0
            elif i == 'hotels':
                df.loc[df[list_cur].iloc[:, 1].isna(), list_cur[1]] = 0
                df.loc[df[list_cur].iloc[:, 2] == 'Москва', list_cur[2]] = 0

            df[i] = 0

            df_cur = df[list_cur].to_numpy().astype(int)

            if len(list_cur) == 4:
                filter_cur = df_cur[:, 3] != 0

                df[i].iloc[filter_cur] = ((df_cur[:, 0] * 0.4 + (df_cur[:, 1] - df_cur[:, 0]) * 0.3 + \
                                           (df_cur[:, 2] - df_cur[:, 1]) * 0.2 + \
                                           (df_cur[:, 3] - df_cur[:, 2]) * 0.1) / df_cur[:, 3])[filter_cur]

            if len(list_cur) == 3:
                filter_cur = df_cur[:, 2] != 0

                df[i].iloc[filter_cur] = ((df_cur[:, 0] * 0.5 + (df_cur[:, 1] - df_cur[:, 0]) * 0.3 + \
                                           (df_cur[:, 2] - df_cur[:, 1]) * 0.2) / df_cur[:, 2])[filter_cur]

            df = df.drop(columns=list_cur)
        return df

    @staticmethod
    def to_cartesian(lat:float, lon:float) -> (float, float):
        '''
        Transforms longtitude and latitude to cartesian system.
        '''
        R = 6371000
        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        return x, y

    def fill_na_with_neib(self, df:pd.DataFrame, feature:str, rad:float=500) -> pd.DataFrame:
        '''
        To fill Nans as avarage in neigbouring objects (currently is NOT using).
        :param df: input DataFrame
        :param feature: feature to fill
        :param rad: radius to look and average
        :return: Dataframe with filled Nans.
        '''
        df['X'], df['Y'] = self.to_cartesian(df['lat'], df['lng'])
        coord_tree = KDTree(df[['X', 'Y']])
        for i, results in tqdm.tqdm(enumerate(coord_tree.query_ball_point(
                df[df[feature].isna()][['X', 'Y']].values, rad)), total=df[df[feature].isna()].shape[0]):
            curr_obj = df['id'].values[results[0]]

            neibs = df['id'].values[results[1:]]

            val_to_fill = df[df['id'].isin(neibs)][feature].mean()

            df.loc[(df['id'] == curr_obj), feature] = val_to_fill
        df = df.drop(columns=['X', 'Y'])
        return df

    def fillna_by_mean(self, df:pd.DataFrame, feature:str, thresh:int=10, by:str='mean') -> pd.DataFrame:
        '''
        Fills Nans by mean over the city, then over the region
        :param df: input DataFrame
        :param feature: feature to fill
        :param thresh: minimum samples in city/region to average over it
        :param by: mean of median averaging
        :return:
        '''
        print('%s City Filling' % feature)
        for city in tqdm.tqdm(df['city'].unique(), total=df['city'].unique().shape[0]):
            city_ind = df[df['city'] == city].index
            if df.loc[city_ind].shape[0] > thresh:
                missed_inds = df.loc[city_ind][df.loc[city_ind][feature].isna()].index
                if by == 'mean':
                    df.loc[missed_inds, feature] = df.loc[city_ind, feature].mean()
                else:
                    df.loc[missed_inds, feature] = df.loc[city_ind, feature].median()

        print('%s Regions Filling' % feature)

        for region in tqdm.tqdm(df['region'].unique(), total=df['region'].unique().shape[0]):
            region_ind = df[df['region'] == region].index
            if df.loc[region_ind].shape[0] > thresh:
                missed_inds = df.loc[region_ind][df.loc[region_ind][feature].isna()].index
                if by == 'mean':
                    df.loc[missed_inds, feature] = df.loc[region_ind, feature].mean()
                else:
                    df.loc[missed_inds, feature] = df.loc[region_ind, feature].median()

        missed_inds = df[df[feature].isna()].index
        print('Left missings: %d' % (missed_inds.shape[0]))
        if missed_inds.shape[0] > 0:
            df.loc[missed_inds, feature] = df[feature].mean()
        return df

    def correct_coords(self, df:pd.DataFrame, thresh:int=100) -> pd.DataFrame:
        '''
        Ensure that city lays inside 2 degrees of lat and lng.
        :param df: input DataFrame
        :param thresh: minimum required record for the city
        :return: corrected DataFrame
        '''
        for city in tqdm.tqdm(df['city'].unique(), total=df['city'].unique().shape[0]):
            city_ind = df[df['city'] == city].index
            if df.loc[city_ind].shape[0] > thresh:
                med_lng, med_lat = df.loc[city_ind]['lng'].median(), df.loc[city_ind]['lat'].median()
                mask = df.loc[city_ind][(df.loc[city_ind]['lng'] < med_lng - 1) |
                                        (df.loc[city_ind]['lng'] > med_lng + 1) |
                                        (df.loc[city_ind]['lat'] < med_lat - 1) |
                                        (df.loc[city_ind]['lat'] > med_lat + 1)].index
                df.loc[mask, 'lng'] = med_lng
                df.loc[mask, 'lat'] = med_lat
        return df

    def cat_features(self, df:pd.DataFrame, features:list) ->pd.DataFrame:
        '''
        One hot encoding
        '''
        for feature in features:
            cat_feature = pd.get_dummies(df[feature])
            df = df.drop(columns=feature)
            df = pd.concat([df, cat_feature], axis=1)
        return df

    def transform(self, fill_features_mean:list=None, fill_features_neib:list=None,
                  cat_features:list=None, drop_cols:list=None):
        '''
        Function that combines all transformations and preprocessing.
        :param fill_features_mean: list of features to fill with city mean
        :param fill_features_neib: list of features to fill with neibours mean
        :param cat_features: list of categorical features
        :param drop_cols: list of columns to drop in the end
        :return:
        '''
        self.subway_cut()
        if self.mode == 'train':
            self.clean_outl()

        self.df_out = self.combine_dist_features(self.df_out)

        if cat_features is not None:
            self.df_out = self.cat_features(self.df_out, cat_features)

        if fill_features_neib is not None:
            for feature in fill_features_neib:
                print('Filling feature %s, missings = %d' % (feature, self.df_out[feature].isna().sum()))
                self.df_out = self.fill_na_with_neib(self.df_out, feature)
                print('Missings = %d left after neib filling' % (self.df_out[feature].isna().sum()))
                self.df_out.loc[self.df_out[feature].isna(), feature] = self.df_out[feature].mean()

        if fill_features_mean is not None:
            for feature in fill_features_mean:
                self.df_out = self.fillna_by_mean(self.df_out, feature)

        self.df_out = self.correct_coords(self.df_out)

        if drop_cols is not None:
            self.df_out = self.df_out.drop(columns=drop_cols)
        return self.df_out


def floor_cleaning(df:pd.DataFrame) -> pd.DataFrame:
    '''
    The function to manually clean errors from floor feature.
    The idea was found in public solution:
     https://github.com/BatyaZhizni/Raifhack-DS/blob/main/notebook/RF%20final%20submission.ipynb
    :param df: DataFrame with floor feature
    :return: DataFrame with cleaned floor feature
    '''
    if 'floor' not in df.columns:
        return df
    df['floor'] = df['floor'].mask(df['floor'] == '-1.0', -1) \
        .mask(df['floor'] == '-2.0', -2) \
        .mask(df['floor'] == '-3.0', -3) \
        .mask(df['floor'] == 'подвал, 1', 1) \
        .mask(df['floor'] == 'подвал', -1) \
        .mask(df['floor'] == 'цоколь, 1', 1) \
        .mask(df['floor'] == '1,2,антресоль', 1) \
        .mask(df['floor'] == 'цоколь', 0) \
        .mask(df['floor'] == 'тех.этаж (6)', 6) \
        .mask(df['floor'] == 'Подвал', -1) \
        .mask(df['floor'] == 'Цоколь', 0) \
        .mask(df['floor'] == 'фактически на уровне 1 этажа', 1) \
        .mask(df['floor'] == '1,2,3', 1) \
        .mask(df['floor'] == '1, подвал', 1) \
        .mask(df['floor'] == '1,2,3,4', 1) \
        .mask(df['floor'] == '1,2', 1) \
        .mask(df['floor'] == '1,2,3,4,5', 1) \
        .mask(df['floor'] == '5, мансарда', 5) \
        .mask(df['floor'] == '1-й, подвал', 1) \
        .mask(df['floor'] == '1, подвал, антресоль', 1) \
        .mask(df['floor'] == 'мезонин', 2) \
        .mask(df['floor'] == 'подвал, 1-3', 1) \
        .mask(df['floor'] == '1 (Цокольный этаж)', 0) \
        .mask(df['floor'] == '3, Мансарда (4 эт)', 3) \
        .mask(df['floor'] == 'подвал,1', 1) \
        .mask(df['floor'] == '1, антресоль', 1) \
        .mask(df['floor'] == '1-3', 1) \
        .mask(df['floor'] == 'мансарда (4эт)', 4) \
        .mask(df['floor'] == '1, 2.', 1) \
        .mask(df['floor'] == 'подвал , 1 ', 1) \
        .mask(df['floor'] == '1, 2', 1) \
        .mask(df['floor'] == 'подвал, 1,2,3', 1) \
        .mask(df['floor'] == '1 + подвал (без отделки)', 1) \
        .mask(df['floor'] == 'мансарда', 3) \
        .mask(df['floor'] == '2,3', 2) \
        .mask(df['floor'] == '4, 5', 4) \
        .mask(df['floor'] == '1-й, 2-й', 1) \
        .mask(df['floor'] == '1 этаж, подвал', 1) \
        .mask(df['floor'] == '1, цоколь', 1) \
        .mask(df['floor'] == 'подвал, 1-7, техэтаж', 1) \
        .mask(df['floor'] == '3 (антресоль)', 3) \
        .mask(df['floor'] == '1, 2, 3', 1) \
        .mask(df['floor'] == 'Цоколь, 1,2(мансарда)', 1) \
        .mask(df['floor'] == 'подвал, 3. 4 этаж', 3) \
        .mask(df['floor'] == 'подвал, 1-4 этаж', 1) \
        .mask(df['floor'] == 'подва, 1.2 этаж', 1) \
        .mask(df['floor'] == '2, 3', 2) \
        .mask(df['floor'] == '7,8', 7) \
        .mask(df['floor'] == '1 этаж', 1) \
        .mask(df['floor'] == '1-й', 1) \
        .mask(df['floor'] == '3 этаж', 3) \
        .mask(df['floor'] == '4 этаж', 4) \
        .mask(df['floor'] == '5 этаж', 5) \
        .mask(df['floor'] == 'подвал,1,2,3,4,5', 1) \
        .mask(df['floor'] == 'подвал, цоколь, 1 этаж', 1) \
        .mask(df['floor'] == '3, мансарда', 3) \
        .mask(df['floor'] == 'цоколь, 1, 2,3,4,5,6', 1) \
        .mask(df['floor'] == ' 1, 2, Антресоль', 1) \
        .mask(df['floor'] == '3 этаж, мансарда (4 этаж)', 3) \
        .mask(df['floor'] == 'цокольный', 0) \
        .mask(df['floor'] == '1,2 ', 1) \
        .mask(df['floor'] == '3,4', 3) \
        .mask(df['floor'] == 'подвал, 1 и 4 этаж', 1) \
        .mask(df['floor'] == '5(мансарда)', 5) \
        .mask(df['floor'] == 'технический этаж,5,6', 5) \
        .mask(df['floor'] == ' 1-2, подвальный', 1) \
        .mask(df['floor'] == '1, 2, 3, мансардный', 1) \
        .mask(df['floor'] == 'подвал, 1, 2, 3', 1) \
        .mask(df['floor'] == '1,2,3, антресоль, технический этаж', 1) \
        .mask(df['floor'] == '3, 4', 3) \
        .mask(df['floor'] == '1-3 этажи, цоколь (188,4 кв.м), подвал (104 кв.м)', 1) \
        .mask(df['floor'] == '1,2,3,4, подвал', 1) \
        .mask(df['floor'] == '2-й', 2) \
        .mask(df['floor'] == '1, 2 этаж', 1) \
        .mask(df['floor'] == 'подвал, 1, 2', 1) \
        .mask(df['floor'] == '1-7', 1) \
        .mask(df['floor'] == '1 (по док-м цоколь)', 1) \
        .mask(df['floor'] == '1,2,подвал ', 1) \
        .mask(df['floor'] == 'подвал, 2', 2) \
        .mask(df['floor'] == 'подвал,1,2,3', 1) \
        .mask(df['floor'] == '1,2,3 этаж, подвал ', 1) \
        .mask(df['floor'] == '1,2,3 этаж, подвал', 1) \
        .mask(df['floor'] == '2, 3, 4, тех.этаж', 2) \
        .mask(df['floor'] == 'цокольный, 1,2', 1) \
        .mask(df['floor'] == 'Техническое подполье', -1) \
        .mask(df['floor'] == '1.2', 1) \
        .astype(float)
    return df