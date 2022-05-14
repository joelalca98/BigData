from operator import length_hint
from matplotlib.pyplot import clf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #Nos permite divide nuestros datos en la parte de train y de test
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('Eopinions.csv') 
#Dentro del DataFrame accedemos a la columna texto y le decimos que cuente las palabras
x = df['text'].str.count('a')
# print(x)
#Para ver el valor medio y la descripción, será más eficiente con palabras largas que cortas
y = df['text'].str.split().str.len().describe()
# print(y)

#Ahora accedemos a la otra columna, Clase. 350 comentarios de Camara y 250 de Autos
z = df['class'].value_counts()
# print(z)

#Valores nulos, los observamos
df.isnull().values.any() #Nos devuelve False porque no hay ningun valor nul, es un dataset bastante bueno

#Ver la suma de valores null por columnas
df.isnull().sum()

#Ver todos los valores nulos sumados
df.isnull().sum().sum()

#ENTRENAMOS
train, test = train_test_split(df, test_size=0.33, random_state=42) #Scikit-learn es un algoritmo predictivo para el analysis de datos

len(train)
# print() #402 --> 66% (Total=600)

len(test)
# print() #198 --> 33%

train['class'].value_counts() #Camera 239 y Auto = 163 ; 402!

train_x = train['text'].to_list() #Lo separamos en listas lo que es el texto y la clase
train_y = train['class'].to_list()

test_x = test['text'].to_list()
test_y = test['class'].to_list()

#Para obtener nuestro modelo utilizamos el modelo de bolsa de palabras (Bag-of-words), cada docuemnto será una bolsa de palabras
# vectorizer = CountVectorizer
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)
train_x_vectors.shape #Nos sale una matriz de 402 (longitud esperada) pero hay 1058 palabras (no esperado)
#Ejemplo de la primera palabra en la posicion 0
train_x_vectors[0]

#Utilizamos el algoritmo de Decision Tree Classifier, basado en un árbol de decisión
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)
#Modelo creado

#Ahora evaluamos el modelo fase 5
k = test_x[4] #Aparentemente es una camara por el texto que aparece 
# print(k)
#Predecimos
print(clf_dec.predict(test_x_vectors[4]))

#DEMO
test_de_prueba = ['I like my new car']
new_test = vectorizer.transform(test_de_prueba)
print(clf_dec.predict(new_test))

#XDD
# plt.figure(figsize=(13,6))
# x.value_counts().plot(kind='bar', color = sns.color_palette("cubehelix"))
# plt.show()




