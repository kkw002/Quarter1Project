from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
def holdout_results(rmodel,xmodel,X,y):
  print('Holdout Random Forest AUC: '+str(metrics.roc_auc_score(y.to_numpy(),rmodel.predict_proba(X)[:,1])))
  print('Generating Plot...\n')
  fpr,tpr,threshholds = metrics.roc_curve(np.array(y),rmodel.predict_proba(X)[:,1])
  plt.plot(fpr,tpr,label='AUC= '+str(metrics.roc_auc_score(y.to_numpy(),rmodel.predict_proba(X)[:,1])))
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend()
  plt.savefig('auc_plots/holdoutrf_auc_plot')
  plt.clf()
  print('Holdout XGB AUC: '+str(metrics.roc_auc_score(y.to_numpy(),xmodel.predict_proba(X)[:,1])))
  print('Generating Plot...\n')
  fpr,tpr,threshholds = metrics.roc_curve(np.array(y),xmodel.predict_proba(X)[:,1])
  plt.plot(fpr,tpr,label='AUC= '+str(metrics.roc_auc_score(y.to_numpy(),xmodel.predict_proba(X)[:,1])))
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend()
  plt.savefig('auc_plots/holdoutxgb_auc_plot')
  plt.clf()
