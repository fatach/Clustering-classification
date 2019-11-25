import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn import svm as SVM
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
import time
from matplotlib import pyplot as plt
import seaborn as sns
time_start = time.time()

data = pd.read_csv('tripadvisor.csv')

#print(data)

#print("valeurs manquantes?:",  data.isnull().values.any())
#print(data.apply(lambda x: sum(x.isnull()),axis=0))

#Recuperation des valeur num
dat= data._get_numeric_data()
x= dat.iloc[:,:10]

#Clustering avc k-means
# build model
kmean =KMeans(n_clusters=5)
#print(KMeans)
kmean.fit(x)
#
notes =kmean.labels_
#print(notes)

#fusion des groupes aux domnnees
data["notes"]=notes
#print(data)
#sauvergarder sous format csv
data.to_csv("./new.csv")
print("-----------------------------------------")
df= pd.read_csv('new.csv')
print(df)
print(df.notes[:2])

print('----Selection of the independante variable vector the features----------')
x = df.iloc[:,[2,11]].values
#print(x)
print('--------Selection of the dependante variable vector-----')
y = df.iloc[:, 12].values
#print(y)

# Visualisation des donnees
y=df.notes[:]
x=dat[:]
tsne = TSNE().fit_transform(x)
print('Temps execution  : {}  seconds' .format(time.time()-time_start))
plt.figure(figsize=(6, 5))
lenX, lenY=data.shape
print(lenX,lenY)
for i in range(lenX):
    plt.scatter(tsne[y == i, 0], tsne[y == i, 1])
#plt.legend()
plt.show()

#Splitting the data into the training set and test set
print('--------------Splitting the data into the training set and test set-----------------------')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size =0.20, random_state = 0)
print('Les dimensions sont--------------------------------------')
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) # here we don't need to fit the  test set

#Fitting the KNN
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=25, p=2)
classifier.fit(x_train, y_train)

#  Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
"""

#Optimisation des parametres
print('****************Optimisation des parametres***************************')
valeurC=[0.0001,0.001,0.01,10,100,1000]
svms=SVM.SVC(gamma='auto')
svms.fit(x_train,y_train)
grid=GridSearchCV(estimator=svms,cv=5,iid=False, param_grid=dict(C=valeurC))
grid.fit(x_train,y_train)

# Les meilleurs parametres
print('The best param',grid.best_params_)

print('+++++++++++++++++++++++++++++++')
#hyper-parameter after tuning
print('____The best estimator____',grid.best_estimator_)
print(grid.best_score_)

#print classification report
print('-------------- classification report-----------------------\n')
grid_predictions = grid.predict(x_test)
print(classification_report(y_test,grid_predictions))


# Fitting SVM methode
classifier = SVM.SVC(kernel='rbf',C=100,gamma='auto', random_state=None)#kernel type :rbf(default) ,linear,poly,sigmoide,precomputed,callable
classifier.fit(x_train, y_train)
print('____Classifier____', classifier)

print('_________________________Prediction_______________________')
y_pred = svms.predict(x_test)
#print(y_pred)

print('_______Confusion_matrix is used to evaluate the correct and the incorrect prediction______')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('___Confusion Matrix____\n',cm)

from sklearn.metrics import accuracy_score
print("____________the accuracy of the model is:",accuracy_score(y_test, y_pred))

print("-----------------Visualising the Training set results-------------------")
sns.set(style="darkgrid")
donneeParGroupe=df.groupby('notes')
print(donneeParGroupe.count())


sns.lineplot(x="User ID", y="notes", data=df);
sns.catplot(x="User ID", y="notes",kind='bar', data=df);
plt.show()

