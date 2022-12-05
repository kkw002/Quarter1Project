import sys
import json
import pandas as pd
sys.path.insert(0, 'src')
from etl import getData,getDataZip
from features import clean_features
from train_model import model_gen
from holdout_model import holdout_results

def main(targets):
    if 'test' in targets:
        data = getData('data/testdata.pkl')
        X,y = clean_features(data)
        model_gen(X,y)
    if 'results' in targets:
        train_set = getData('data/forStudents.pkl.zip')
        holdout_set = getData('data/holdout_final.pkl.zip')
        train_X,train_y = clean_features(train_set)
        holdout_X,holdout_y = clean_features(holdout_set)
        randmodel,xgbmodel = model_gen(train_X,train_y)
        holdout_results(randmodel,xgbmodel,holdout_X,holdout_y)
    return 
if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
