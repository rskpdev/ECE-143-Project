#!/usr/bin/env python
# coding: utf-8

# In[137]:


import pandas as pd
import vaex
import numpy as np
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing, svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder,LabelEncoder, PolynomialFeatures, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# In[138]:


df = pd.read_csv('heart.csv')
df.head()


# In[139]:


le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)


# In[140]:


corr = df.corr(method = 'pearson')# std covariance
ax = sns.heatmap(corr,annot=True)
plt.title('Correlation Matrix')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
plt.show()


# In[141]:


X = df.drop(['FastingBS','Cholesterol','RestingBP','HeartDisease'],axis = 1)


# In[142]:


y = df['HeartDisease']


# In[143]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[144]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# In[145]:


# Logistic Regression
model = LogisticRegression(verbose = 1)
model.fit(X_train_scaled,y_train)


# In[146]:


y_pred = model.predict(X_test_scaled)
y_test = np.array(y_test)
y_pred


# In[147]:


model.score(X_test_scaled,y_test)
accuracy_score(y_pred,y_test)


# In[148]:


penalty = ["l1", "l2", "elasticnet"]
l1_ratio = np.linspace(0, 1, 20)
C = np.logspace(0, 10, 20)

param_grid = {"penalty" : penalty,
             "l1_ratio" : l1_ratio,
             "C" : C}


# In[149]:


LR_Model = LogisticRegression(solver = 'saga',max_iter=5000, class_weight = "balanced")
LR_Model = GridSearchCV(LR_Model, param_grid = param_grid)


# In[150]:


LR_Model.fit(X_train_scaled,y_train)


# In[151]:


y_pred = LR_Model.predict(X_test_scaled)
y_pred


# In[152]:


y_test


# In[153]:


LR_Model.score(X_test_scaled,y_test)


# In[154]:


accuracy_score(y_pred,y_test)


# In[155]:


# KMeans


# In[156]:


from sklearn.cluster import KMeans 


# In[157]:


model = KMeans(n_clusters = 2)


# In[158]:


model.fit(X_train_scaled)


# In[159]:


model.cluster_centers_


# In[160]:


model.labels_


# In[161]:


model.predict(X_test_scaled)


# In[162]:


from sklearn import metrics


# In[163]:


score = metrics.accuracy_score(y_test,model.predict(X_test_scaled))


# In[164]:


score


# In[165]:


y = model.predict(X_test_scaled)


# In[ ]:





# In[ ]:





# In[167]:


#SVM Model


# In[168]:


from sklearn.svm import SVR
from sklearn.svm import SVC
svm_model = SVC(random_state = 49)
svm_model.fit(X_train_scaled,y_train)


# In[169]:


svm_model.predict(X_test_scaled)


# In[170]:


svm_model.score(X_test_scaled,y_test)


# In[171]:


param_grid = {'C': [0.1,1, 10, 100, 1000],
              'gamma': ["scale", "auto", 1,0.1,0.01,0.001,0.0001],
              'kernel': ['rbf', 'linear']}


# In[172]:


SVM_grid_model = SVC(random_state=42)
SVM_grid_model = GridSearchCV(SVM_grid_model, param_grid, verbose=3, refit=True)


# In[173]:


SVM_grid_model.fit(X_train_scaled, y_train)


# In[176]:


y_pred = SVM_grid_model.predict(X_test_scaled)


# In[177]:


SVM_grid_model.score(X_test_scaled,y_test)


# In[178]:


accuracy_score(y_pred,y_test)


# In[179]:


#KNN Model


# In[180]:


from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsRegressor(n_neighbors=1, algorithm="kd_tree")
KNN_model.fit(X_train_scaled, y_train)
KNN_model.predict(X_train_scaled)


# In[181]:


KNN_model.predict(X_test_scaled)


# In[182]:


np.array(y_test)


# In[183]:


from sklearn.metrics import accuracy_score
knn_acc = accuracy_score(y_test, KNN_model.predict(X_test_scaled))


# In[184]:


knn_acc


# In[185]:


k_values= range(1, 30)
param_grid = {"n_neighbors": k_values, "p": [1, 2], "weights": ['uniform', "distance"]}


# In[186]:


KNN_grid_model = KNeighborsClassifier()


# In[187]:


KNN_grid_model = GridSearchCV(KNN_grid_model, param_grid, cv=10, scoring='accuracy')


# In[188]:


KNN_grid_model.fit(X_train_scaled, y_train)


# In[189]:


KNN_grid_model.predict(X_test_scaled)


# In[190]:


y_pred =KNN_grid_model.predict(X_test_scaled)


# In[191]:


accuracy_score(y_pred,y_test)


# In[192]:


#Decision Tree


# In[193]:


from sklearn.tree import DecisionTreeClassifier


# In[194]:


DT_model = DecisionTreeClassifier(class_weight="balanced", random_state=63)
DT_model.fit(X_train_scaled, y_train)


# In[195]:


y_pred = DT_model.predict(X_test_scaled)


# In[196]:


y_pred


# In[197]:


y_train_pred = DT_model.predict(X_train_scaled)


# In[198]:


acc_score = accuracy_score(y_pred,y_test)


# In[199]:


acc_score


# In[200]:


param_grid = {"splitter":["best", "random"],
              "max_features":[None, 3, 5, 7],
              "max_depth": [None, 4, 5, 6, 7, 8, 9, 10],
              "min_samples_leaf": [2, 3, 5],
              "min_samples_split": [2, 3, 5, 7, 9, 15]}


# In[202]:


#DT_grid_model = DecisionTreeClassifier(class_weight = "balanced", random_state=63)
DT_grid_model = GridSearchCV(estimator=DT_model,
                            param_grid=param_grid,
                            scoring='recall',
                            n_jobs = -1, verbose = 2)


# In[203]:


DT_grid_model.fit(X_train_scaled,y_train)


# In[204]:


y_pred = DT_grid_model.predict(X_test_scaled)


# In[205]:


accuracy_score(y_pred,y_test)


# In[ ]:





# In[206]:


# random Forest


# In[207]:


from sklearn.ensemble import RandomForestClassifier


# In[208]:


RF_model = RandomForestClassifier(class_weight="balanced", random_state=101)
RF_model.fit(X_train_scaled, y_train)
y_pred = RF_model.predict(X_test_scaled)
y_train_pred = RF_model.predict(X_train_scaled)


# In[209]:


acc_score_rf = accuracy_score(y_pred,y_test)


# In[210]:


acc_score_rf


# In[211]:


param_grid = {'n_estimators':[50, 100, 300],
             'max_features':[2, 3, 4],
             'max_depth':[3, 5, 7, 9],
             'min_samples_split':[2, 5, 8]}


# In[212]:


RF_grid_model = GridSearchCV(estimator=RF_model, 
                             param_grid=param_grid, 
                             scoring = "recall", 
                             n_jobs = -1, verbose = 2)


# In[213]:


RF_grid_model.fit(X_train_scaled,y_train)


# In[214]:


RF_grid_model.predict(X_test_scaled)


# In[215]:


y_pred = RF_grid_model.predict(X_test_scaled)


# In[216]:


accuracy_score(y_pred,y_test)


# In[ ]:




