import pandas as pd
def clean_features(df):
  encoded_X= pd.get_dummies(df,columns=['acquisition_type','snapshot_type','channel'])
  encoded_X['vantage3_score'].fillna(encoded_X['vantage3_score'].mean(), inplace=True)
  #removing these columns specifically
  X=encoded_X.loc[:, ~encoded_X.columns.isin(['bad','bad_balance','vintage','state_code','bad_v2','evaluation_dt'])]
  y = encoded_X['bad']
  X=X.replace(np.NaN,-999)
  y=y.replace(np.NaN,-2)
  return X,y
