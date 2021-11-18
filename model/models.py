from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def Logistic_Regression_Model(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, Y_train)
    y_bar = model.predict(X_test)
    acc = accuracy_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=model.classes_)
    plt.figure()
    dis.plot()
    plt.show()
    return acc, confusion


def DecisionTree_Model(X_train, Y_train, X_test, Y_test):
    model = DecisionTreeClassifier(class_weight="balanced")
    model.fit(X_train, Y_train)
    y_bar = model.predict(X_test)
    acc = accuracy_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=model.classes_)
    plt.figure()
    dis.plot()
    plt.show()
    return acc, confusion


def RandomForest_Model(X_train, Y_train, X_test, Y_test):
    model = RandomForestClassifier(class_weight="balanced")
    model.fit(X_train, Y_train)
    y_bar = model.predict(X_test)
    acc = accuracy_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=model.classes_)
    plt.figure()
    dis.plot()
    plt.show()
    return acc, confusion


def SVM_Model(X_train, Y_train, X_test, Y_test):
    model = SVC()
    model.fit(X_train, Y_train)
    y_bar = model.predict(X_test)
    acc = accuracy_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=model.classes_)
    plt.figure()
    dis.plot()
    plt.show()
    return acc, confusion


def KNN_Model(X_train, Y_train, X_test, Y_test, n):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, Y_train)
    y_bar = model.predict(X_test)
    acc = accuracy_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=model.classes_)
    plt.figure()
    dis.plot()
    plt.show()
    return acc, confusion
