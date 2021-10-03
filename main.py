import argparse
import pandas as pd
from model.train import train
from model.predict import predict
from model.gettestsplit import get_test_split
from model.grids import grid_zeros, grid_ones

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        '''
        Raifhack solution from Crystal team.
        To learn new model run in your terminal:
            
        ''',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('--mode', '-m', type = str, dest = 'mode', required=True, help = 'Mode of running: train or predict')
    parser.add_argument('--train_data', '-td', type=str, dest='traind', help='path to training data')
    parser.add_argument('--predict_data', '-pd', type=str, dest='predictd', help='path to testing data')
    parser.add_argument("--model_path", '-mp', type=str, dest='mp', help='path to pickle model (save or use - '
                                                                         'depends on the mode)')
    parser.add_argument('--output', '-o', type=str, dest='o', help='path to output csv file')

    return parser.parse_args()

def main():
    args = vars(parse_args())
    mode = args['mode']
    if (mode != 'train') and (mode != 'predict'):
        raise ValueError('Unknown mode set! Set either train or predict!')

    if mode =='train':
        if args['traind'] is not None:
            data_train = pd.read_csv(args['traind'])
            data_train.loc[(data_train['region'] == 'Алтай'), 'region'] = 'Алтайский край'
        else:
            raise  ValueError('Path to train data is not set!')
        try:
            test_inds = get_test_split('test_inds.pkl', data_train = data_train)
            if args['mp'] is not None:
                save_path = args['mp']
            else:
                save_path = 'model.pkl'
            train(data_train, test_inds, grid_zeros, grid_ones, savepath = save_path)
        except:
            raise IOError('Some error has happend while training!')

    elif mode == 'predict':
        if args['predictd'] is not None:
            data_test = pd.read_csv(args['predictd'])
        else:
            raise  ValueError('Path to train data is not set!')
        try:
            if args['mp'] is None:
                raise ValueError('Path to the model is not indicated!')
            if args['o'] is not None:
                outname = args['o']
            else:
                outname = 'submit.csv'
            predict(data_test, modelname=args['mp'], outname=outname)
        except:
            raise IOError('Some error has happend while training!')

if __name__ == "__main__":
    main()