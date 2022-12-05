import pandas as pd
def clean_features(df):
  encoded_X= pd.get_dummies(df,columns=['acquisition_type','snapshot_type','channel'])
  encoded_X['vantage3_score'].fillna(encoded_X['vantage3_score'].median(), inplace=True)
  #removing these columns specifically
  encoded_X= encoded_X[(encoded_X['all0000'].isnull()==False)&(encoded_X['bad']!=-1)&(encoded_X['bad'].isnull()==False)]
  X = encoded_X.loc[:, ~encoded_X.columns.isin(['net_spend','current_balance','bad','bad_balance','vintage','state_code','bad_v2','evaluation_dt','all9230','all9240','all9249','all9280'])]
  X[X.columns[4:823]]=X[X.columns[4:823]].fillna(X[X.columns[4:823]].median())
  #prediction var in this case is bad
  y = encoded_X['bad']
  return X,y
