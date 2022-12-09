import sklearn
from sklearn.svm import SVC

def train(dataset, train_data, train_label, test_data, test_label):
    if dataset == "face":
        clf = SVC(kernel='linear')
        new_train_data = train_data.reshape(752, 4200)
        clf.fit(new_train_data, train_label)
        new_test_data = test_data.reshape(150, 4200)
        y_pred = clf.predict(new_test_data)

        print("Accuracy for Face dataset:", sklearn.metrics.accuracy_score(test_label, y_pred)*100)

    elif dataset == "digit":
        clf = SVC(kernel='linear')
        new_train_data = train_data.reshape(6000, 784)
        clf.fit(new_train_data, train_label)
        new_test_data = test_data.reshape(1000, 784)
        y_pred = clf.predict(new_test_data)

        print("Accuracy for Digit dataset:", sklearn.metrics.accuracy_score(test_label, y_pred) * 100)
