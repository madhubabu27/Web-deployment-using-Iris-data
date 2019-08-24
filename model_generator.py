from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 
import pickle 

iris = datasets.load_iris() 
validation_size = 0.20 
seed = 100 
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, 
                                                    iris.target, 
                                                    test_size=validation_size, 
                                                    random_state=seed) 

knn = KNeighborsClassifier() 
knn.fit(X_train, Y_train) 

with open('models/iris_classifier_model.pk', 'wb') as model_file: 
    pickle.dump(knn, model_file)