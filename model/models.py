import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PCA import PCA_BestFeatures
from sklearn.model_selection import GridSearchCV


def Logistic_Regression_Model(X_train, Y_train, X_test, Y_test):
    l1_ratio = np.linspace(0, 1, 20)
    C = np.logspace(0, 10, 20)
    param_grid = {
        "l1_ratio": l1_ratio,
        "C": C}
    LR_Model = LogisticRegression(solver='saga', max_iter=5000, class_weight="balanced")
    LR_Model = GridSearchCV(LR_Model, param_grid=param_grid)
    LR_Model.fit(X_train, Y_train)
    y_bar = LR_Model.predict(X_test)
    pre_score_lr = precision_score(Y_test, y_bar)
    acc_lr = accuracy_score(Y_test, y_bar)
    rec_score_lr = recall_score(Y_test, y_bar)
    f1_lr = f1_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=LR_Model.classes_)
    dis.plot()
    plt.title("confusion matrix for logistic regression")
    plt.show()
    PCA_BestFeatures(X_test, Y_test, y_bar)
    return [acc_lr, pre_score_lr, rec_score_lr, f1_lr]


def DecisionTree_Model(X_train, Y_train, X_test, Y_test):
    DT_model = DecisionTreeClassifier(class_weight="balanced", random_state=63)

    param_grid = {"splitter": ["best", "random"],
                  "max_features": [None, 3, 5, 7],
                  "max_depth": [None, 4, 5, 6, 7, 8, 9, 10],
                  "min_samples_leaf": [2, 3, 5],
                  "min_samples_split": [2, 3, 5, 7, 9, 15]}

    DT_grid_model = GridSearchCV(estimator=DT_model,
                                 param_grid=param_grid,
                                 scoring='recall',
                                 n_jobs=-1, verbose=2)
    DT_grid_model.fit(X_train, Y_train)
    y_bar = DT_grid_model.predict(X_test)
    pre_score_dt = precision_score(Y_test, y_bar)
    acc_dt = accuracy_score(Y_test, y_bar)
    rec_score_dt = recall_score(Y_test, y_bar)
    f1_dt = f1_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=DT_grid_model.classes_)
    dis.plot()
    plt.title("confusion matrix for decision tree")
    plt.show()
    PCA_BestFeatures(X_test, Y_test, y_bar)
    return [acc_dt, pre_score_dt, rec_score_dt, f1_dt]


def RandomForest_Model(X_train, Y_train, X_test, Y_test):
    RF_model = RandomForestClassifier(class_weight="balanced", random_state=101)
    param_grid = {'n_estimators': [50, 100, 300],
                  'max_features': [2, 3, 4],
                  'max_depth': [3, 5, 7, 9],
                  'min_samples_split': [2, 5, 8]}
    RF_grid_model = GridSearchCV(estimator=RF_model,
                                 param_grid=param_grid,
                                 scoring="recall",
                                 n_jobs=-1, verbose=2)

    RF_grid_model.fit(X_train, Y_train)
    y_bar = RF_grid_model.predict(X_test)
    pre_score_rf = precision_score(Y_test, y_bar)
    acc_rf = accuracy_score(Y_test, y_bar)
    rec_score_rf = recall_score(Y_test, y_bar)
    f1_rf = f1_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=RF_grid_model.classes_)
    dis.plot()
    plt.title("confusion matrix for random forest")
    plt.show()
    PCA_BestFeatures(X_test, Y_test, y_bar)
    return [acc_rf, pre_score_rf, rec_score_rf, f1_rf]


def SVM_Model(X_train, Y_train, X_test, Y_test):
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': ["scale", "auto", 1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear']}

    SVM_grid_model = SVC(random_state=42)
    SVM_grid_model = GridSearchCV(SVM_grid_model, param_grid, verbose=3, refit=True)
    SVM_grid_model.fit(X_train, Y_train)
    y_bar = SVM_grid_model.predict(X_test)
    pre_score_svm = precision_score(Y_test, y_bar)
    acc_svm = accuracy_score(Y_test, y_bar)
    rec_score_svm = recall_score(Y_test, y_bar)
    f1_svm = f1_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=SVM_grid_model.classes_)
    dis.plot()
    plt.title("confusion matrix for support vector machine")
    plt.show()
    PCA_BestFeatures(X_test, Y_test, y_bar)
    return [acc_svm, pre_score_svm, rec_score_svm, f1_svm]


def KNN_Model(X_train, Y_train, X_test, Y_test):
    k_values = range(1, 30)
    param_grid = {"n_neighbors": k_values, "p": [1, 2], "weights": ['uniform', "distance"]}
    KNN_grid_model = KNeighborsClassifier()
    KNN_grid_model = GridSearchCV(KNN_grid_model, param_grid, cv=10, scoring='accuracy')
    KNN_grid_model.fit(X_train, Y_train)
    y_bar = KNN_grid_model.predict(X_test)
    pre_score_knn = precision_score(Y_test, y_bar)
    acc_knn = accuracy_score(Y_test, y_bar)
    rec_score_knn = recall_score(Y_test, y_bar)
    f1_knn = f1_score(Y_test, y_bar)
    confusion = confusion_matrix(Y_test, y_bar)
    dis = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=KNN_grid_model.classes_)
    dis.plot()
    plt.title("confusion matrix for K nearest neighborhood")
    plt.show()
    PCA_BestFeatures(X_test, Y_test, y_bar)
    return [acc_knn, pre_score_knn, rec_score_knn, f1_knn]


def plot(X_train, Y_train, X_test, Y_test):
    import plotly.graph_objects as go
    fig = go.Figure()
    metrics = {'logistic_regression': Logistic_Regression_Model(X_train, Y_train, X_test, Y_test),
               'Decision_tree': DecisionTree_Model(X_train, Y_train, X_test, Y_test),
               'Random_forest': RandomForest_Model(X_train, Y_train, X_test, Y_test),
               'SVM': SVM_Model(X_train, Y_train, X_test, Y_test),
               'KNN': KNN_Model(X_train, Y_train, X_test, Y_test)}
    index = ['accuracy', 'precision', 'recall score', 'f1 score']
    df = pd.DataFrame(metrics)
    df = df.set_index(pd.Index(index))
    fig.add_trace(go.Scatterpolar(
        r=df['logistic_regression'].values,
        theta=index,
        fill='toself',
        name='logistic_regression'
    ))
    fig.add_trace(go.Scatterpolar(
        r=df['Decision_tree'].values,
        theta=index,
        fill='toself',
        name='Decision_tree'
    ))
    fig.add_trace(go.Scatterpolar(
        r=df['Random_forest'].values,
        theta=index,
        fill='toself',
        name='Random_forest'
    ))
    fig.add_trace(go.Scatterpolar(
        r=df['SVM'].values,
        theta=index,
        fill='toself',
        name='SVM'
    ))
    fig.add_trace(go.Scatterpolar(
        r=df['KNN'].values,
        theta=index,
        fill='toself',
        name='KNN'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.75, 1]
            )),
        showlegend=True
    )
    fig.show()

    return df
