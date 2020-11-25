import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def print_confusionmatrix(true_test,preds_test):
  #y_test = np.argmax(y_pred)
  #y_pred = np.argmax(y_pred)

  df = tm.DataFrame(confusion_matrix(true_test,preds_test))
  plt.figure(figsize = (10,10))
  sns.heatmap(df,annot=True)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  #plt.show()
  plt.savefig('figure0.png')

  acc_score = accuracy_score(true_test,preds_test)
  print("acc_score: ",acc_score)
  print("error rate: ",1-acc_score)

  return 
