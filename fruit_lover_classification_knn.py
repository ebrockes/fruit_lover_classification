import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier


#### color score
# red - 0.85 to 1.00
# orange - 0.75 to 0.85
# yellow - 0.65 to 0.75
# green - 0.45 to 0.65
# ...

fruits = pd.read_csv('fruit_data_with_colors.txt')
print(fruits.head())

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print(lookup_fruit_name)

X = fruits[['mass','width','height','color_score']]
y = fruits['fruit_label']

#feature space
fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), title='Box Plot for each input variable')
plt.savefig('fruits_box')
plt.show()

# relations - features
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('fruits_scatter_matrix')
plt.show()


#75 25 partitioning
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3 dimensional plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()

#create classifier object 
#knn = KNeighborsClassifier(n_neighbors = 5)
knn = KNeighborsClassifier(algorithm = 'auto', leaf_size=10, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')

#train the classifier
knn.fit(X_train, y_train)

#accuracy of the classifier
knn.score(X_test, y_test)

print("############ Preditions")

#Use the trained k-NN classifier model to classify new, previously unseen objects
fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.9]])
print(lookup_fruit_name[fruit_prediction[0]])

fruit_prediction = knn.predict([[100, 6.3, 8.5, 0.1]])
print(lookup_fruit_name[fruit_prediction[0]])