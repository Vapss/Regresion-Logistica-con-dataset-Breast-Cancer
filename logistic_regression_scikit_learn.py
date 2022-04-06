
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
df = pd.read_csv('breast-cancer.csv', sep=',', engine='python')

X = df.drop(['id','diagnosis'],axis=1).values   
y = df['diagnosis'].values

plt.scatter(X,y)
	
#Separa el corpus cargado en el DataFrame en el 90% para entrenamiento y el 10% para pruebas
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, shuffle = True, random_state=0)
	
clf = LogisticRegression()
#~ ##~ clf = LogisticRegression(solver = 'liblinear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print ('Clase real{}\nClase predicha{}'.format(y_test, y_pred))
print (X_test)

y_pred_proba = clf.predict_proba(X_test)
print (y_pred_proba)
#~ plt.show()
