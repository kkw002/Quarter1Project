
import pandas as pd

def getData(fp):
  if 'zip' in fp:
    data = pd.read_pickle(fp, compression='zip')
  else:
    data = pd.read_pickle(fp)
  return data
