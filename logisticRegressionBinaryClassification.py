import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split 


#### color score
# red - 0.85 to 1.00
# orange - 0.75 to 0.85
# yellow - 0.65 to 0.75
# green - 0.45 to 0.65
# ...

fruits = pd.read_csv('fruit_data_with_colors.txt')

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print(lookup_fruit_name)

X = fruits[['mass','width','height','color_score']]
y = fruits['fruit_label'].values
y[y == 1] = 1
y[y != 1] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(X, y)

p_pred = model.predict_proba(X)
y_pred = model.predict(X)
score_ = model.score(X, y)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print('# classification report')
print(report)
print('# confusion_matrix')
print(conf_m)
print('# _score')
print(score_)

logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Apple identification (LogisticRegression)')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()