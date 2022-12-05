from sklearn import metrics
def holdout_results(rmodel,xmodel,X,y):
  print('Holdout Random Forest AUC: '+str(metrics.roc_auc_score(y.to_numpy(),rmodel.predict_proba(X)[:,1])))
  print('Holdout XGB AUC: '+str(metrics.roc_auc_score(y.to_numpy(),xmodel.predict_proba(X)[:,1])))
