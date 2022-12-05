
import pandas as pd

def getData(fp):
  data = pd.read_pickle(fp)
  return data
def getDataZip(fp):
  data = pd.read_pickle(fp, compression='zip')
  return data
