import numpy as np
from collections import Counter

####Sample Data
x_val= np.array([6,8,3,1,5,7,9,7,4,4,6,8,3,1])
y_val = np.array([6,4,2,6,7,8,3,2,6,7,8,5,2,1])
labels =  ["blue", "green", "red", "purple", "blue", "green", "red", "purple",
           "blue", "green", "red", "purple", "red", "purple"]

def manhattan_distance(x,x_test,y,y_test):
    xdist  = np.sum(np.abs(x - x_test))
    ydist =np.sum(np.abs(y - y_test))
    return xdist+ydist

def simple_knn(x_test,y_test,x,y,labels):


    dist = []
    for x,y in zip(x,y):
        dist.append(manhattan_distance(x,x_test,y,y_test))

    sorted_dist = sorted(dist)
    pred_label = {}
    k = 2
    for label in range(0,len(labels)):
        neighbours = sorted_dist[:k]
        for i in neighbours:
            for j in range(0,len(dist)):
                if i  == dist[j]:
                    if labels[j] in pred_label:

                        pred_label[labels[j]] += 1
                    else:
                        pred_label[labels[j]] = 1

    pred_label = max(pred_label, key=pred_label.get)
    print(pred_label)


    return  pred_label






simple_knn(4,1,x_val,y_val,labels)


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Provided data
x_val = np.array([6, 8, 3, 1, 5, 7, 9, 7, 4, 4, 6, 8, 3, 1])
y_val = np.array([6, 4, 2, 6, 7, 8, 3, 2, 6, 7, 8, 5, 2, 1])
labels = ["blue", "green", "red", "purple", "blue", "green", "red", "purple",
          "blue", "green", "red", "purple", "red", "purple"]

# Combine x_val and y_val into feature vectors
feature_vectors = np.column_stack((x_val, y_val))

# Initialize the KNN classifier with k=3
knn_classifier = KNeighborsClassifier(n_neighbors=2)

# Train the classifier
knn_classifier.fit(feature_vectors, labels)

# New data point to predict
new_data_point = np.array([[4, 1]])

# Predict the label for the new data point
predicted_label = knn_classifier.predict(new_data_point)

print("Predicted label:", predicted_label[0])


