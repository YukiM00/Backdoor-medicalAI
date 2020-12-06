import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas.util.testing as tm
import seaborn as sns

labels = ['normal','pneumonia']

def make_confusionmatrix(true_test, preds_test, file_name):
  df = tm.DataFrame(confusion_matrix(true_test,preds_test))
  plt.figure(figsize = (10,10))
  sns.heatmap(df,annot=True,
                fmt='.0f',
                annot_kws={'size': 15},
                yticklabels=labels,
                xticklabels=labels)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  #plt.show()
  plt.savefig(file_name)
  acc_score = accuracy_score(true_test,preds_test)
  print("error rate: ",1 - acc_score)
