import sys
import json
import pandas as pd
sys.path.insert(0, 'src')
from etl import getData
from features import clean_features
from train_model import rfModel

def main(targets):
    data_config = json.load(open('config/data-params.json'))
    if 'test' in targets:
      data = getData('test/testdata.pkl')
      X,y = clean_features(data)
      rfModel(X,y)
    if 'results' in targets:
      
return 
if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
