from sklearn import tree, neighbors, svm, naive_bayes, metrics

# Classifiers
classifiers = [
    tree.DecisionTreeClassifier(),
    neighbors.KNeighborsClassifier(),
    svm.SVC(),
    naive_bayes.GaussianNB(),
]

clf_names = [
    'decision_tree',
    'k_neighbors',
    'svc',
    'gaussian_naive_bayes'
]

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# CHALLENGE - ...and train them on our data

for i in range(len(classifiers)):
    classifiers[i] = classifiers[i].fit(X, Y)

# Predictions

for i in range(len(classifiers)):
    pred = []
    for data in X:
        pred.append(classifiers[i].predict([data]))

    print(f"Accuracy of {clf_names[i]}: {metrics.accuracy_score(Y, pred)}")

# CHALLENGE compare their results and print the best one!

