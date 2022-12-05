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
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        with open('config/holdoutdata_params.json') as fh:
            holddata_cfg = json.load(fh)
        train_set = getDataZip(**data_cfg)
        holdout_set = getDataZip(**holddata_cfg)
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
