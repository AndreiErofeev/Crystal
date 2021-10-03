# RaifHack solution by Crystal team

by Andrei Erofeev, Evgenii Kanin and Mikhail Doroshenko

to run training set --mode to train and indicate path to .csv after --train_data, for example:

python main.py --mode train --train_data ./data/train.csv

to run predict set --mode to predict, indicate path to traied model in --model_path and path to data after --predict_data, for example:

python main.py --mode predict --predict_data ./data/test.csv --model_path ./model.pkl
