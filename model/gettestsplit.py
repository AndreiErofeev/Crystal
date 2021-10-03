import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def get_test_split(path:str, DUMP:int = 1, data_train:pd.DataFrame = None, SEED:int = 69) -> dict:
    if DUMP == 1:
        _, test_inds_tp_0 = train_test_split(data_train[data_train['price_type'] == 0].index, test_size=0.1,
                                             stratify=data_train[data_train['price_type'] == 0]['region'],
                                             random_state=SEED)

        df_all_pr_1 = data_train[data_train['price_type'] == 1]
        df_all_pr_1.loc[(df_all_pr_1['region'] == 'Ставропольский край'), 'region'] = 'Краснодарский край'

        _, test_inds_tp_1 = train_test_split(df_all_pr_1.index, test_size=1500,
                                             stratify=df_all_pr_1['region'])

        test_inds = {'pt_0': data_train.loc[test_inds_tp_0]['id'], 'pt_1': data_train.loc[test_inds_tp_1]['id']}

        with open(path, 'wb') as f:
            pickle.dump(test_inds, f)
    else:
        with open(path, 'rb') as f:
            test_inds = pickle.load(f)
    return test_inds