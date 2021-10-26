from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# split dataset into testing and training
X_train, X_test, y_train, y_test = train_test_split()

# instantiate the KNN class
clf = KNeighborsClassifier(n_neighbors=3)