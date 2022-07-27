"""
Caso de estudio Marketing

Elaborado por: 

Laura Rojas
Cristhian Guzman
Nicolas Florez

"""

## Librerias
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3 as sql
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sklearn

"""
Solución

Crear una lista o varias de recomendaciones para cada usario, ordenada por el rating,
ademas el usuario pueda filtrar por genero u año de la pelicula o por temporada del año.

Cuando un usuario nuevo se registre, solicitarle info de generos favoritos,
y de esta forma crear una lista para recomendarle

"""

"""
Problema Negocio
Con el objetivo de que estos tengan una mejor experiencia y esto permita mejorar 
su fidelización y recomendación a nuevos clientes. 

Problema analitico
Crear un algoritmo de recomendaciones de peliculas, basado en un filtrado colaborativo

"""


## Carga Base de datos

conn= sql.connect("db_movies")

df_movies=pd.read_sql("select * from movies", conn)

df_users = pd.read_sql("select *,datetime(timestamp,'unixepoch') as date from ratings", conn)

df_movies
df_users

### Revision de los datos

df_users['userId'].unique().shape[0]
df_movies['movieId'].unique().shape[0]

df_users.info()
df_movies.info()

#cambiar la separacion del genero
df_movies['genres'] = df_movies.genres.str.split('|')

#Formato fecha date
df_users['date']= pd.to_datetime(df_users['date']) 

### Eliminar la columna  timstamp
df_users.drop(['timestamp'], axis=1,inplace=True)

#Eliminar hora de la columna date en userID
df_users['date'] = pd.to_datetime(df_users['date']).dt.date

#Separar genero en columnas
#Para cada fila del marco de datos, iterar la lista de géneros y colocar un 1 en la columna que corresponda
for index, row in df_movies.iterrows():
    for genre in row['genres']:
        df_movies.at[index, genre] = 1
#Completar los valores NaN con 0 para mostrar que una película no tiene el género de la columna
df_movies = df_movies.fillna(0)
df_movies.head()

#Eliminar la columna genero
df_movies.drop(['genres'], axis=1,inplace=True)

#creamos una columna nueva que se llamará year
df_movies['year'] = df_movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Quitando los paréntesis 
df_movies['year'] = df_movies.year.str.extract('(\d\d\d\d)',expand=False)
#Eliminar los años de la columna "título" y el espacio
df_movies['title'] = df_movies.title.str.replace(' (\(\d\d\d\d\))', '')


### ANALISIS EXPLORATORIO

df_movies.info()
df_users.info()

df_movies
df_users
#conteo de peliculas por año
pd.crosstab(index=df_movies['year'], columns=' count ')
pd.crosstab(index=df_users['date'], columns=' count ')

#las personas empezaron a llevar registro de la calificiación desde 1996
#las bases de las peliculas van hasta el 2018

#Conteo de calificación 
pd.crosstab(index=df_users['rating'], columns=' count ')


# Matriz peliculas ID y rating
pd.crosstab(index=df_users['rating'], columns=df_users['movieId'], margins=True)

# Matriz userID y rating
pd.crosstab(index=df_users['rating'], columns=df_users['userId'], margins=True)

"""Las calidicaciones reflejan la cantidad de vistas que ha tenido una pelicula"""



# Matriz de usuario y peliculas con rating, llenando los vacios con 0
pd.pivot_table(df_users, values='rating', index='userId', columns='movieId').fillna(0)

# Union bases de datos

df = df_users.merge(df_movies, on = 'movieId', how = 'left')
df

# Peliculas mas veces calificadas
df.groupby(['movieId','title','year'])['rating'].count().sort_values(ascending=False).head()
# Peliculas menos veces calificadas
df.groupby(['movieId','title','year'])['rating'].count().sort_values(ascending=False).tail()

""" La fecha de la pelicula no debe ser tan determinante para calificarla mas que otra"""

df['month_rating'] = pd.to_datetime(df_users['date']).dt.month
df['year_rating'] = pd.to_datetime(df_users['date']).dt.year

# Cantidad de calificaciones por mes del año
df.groupby(['month_rating'])['rating'].count().sort_values(ascending=False)
# Cantidad de calificaciones por año
df.groupby(['year_rating'])['rating'].count().sort_values(ascending=False)
# Cantidad de calificaciones por año de la pelicula
df.groupby(['year'])['rating'].count().sort_values(ascending=False)

### ------------------ALGORITMOS----------------

#---------filtro colaborativo
#---------basado en usuario
#---------metodo: distancia por coseno de los vectores


#matriz en la que cruzamos todos los usuarios con todos las peliculas
matriz = pd.pivot_table(df_users, values='rating', index='userId', columns='movieId').fillna(0)

#porcentaje de sparcity
ratings = matriz.values
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print('Sparsity: {:4.2f}%'.format(sparsity))

#conjunto de entrenamiento y de prueba
ratings_train, ratings_test = train_test_split(ratings, test_size = 0.33, random_state=42)
print(ratings_train.shape)
print(ratings_test.shape)

#matriz la similitud entre usuarios
sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings)
print(sim_matrix.shape)

#ver similud graficamente
plt.imshow(sim_matrix);
plt.colorbar()
plt.show()

#separar las filas y columnas de train y test
sim_matrix_train = sim_matrix[0:408,0:408]
sim_matrix_test = sim_matrix[408:610,408:610]
users_predictions = sim_matrix_train.dot(ratings_train) / np.array([np.abs(sim_matrix_train).sum(axis=1)]).T

#ver recomendaciones entre todas las peliculas y usuarios graficamente
plt.rcParams['figure.figsize'] = (20.0, 5.0)
plt.imshow(users_predictions);
plt.colorbar()
plt.show()

#ejemplo de prediccón de usuario
usuario_ver = 0 #data.iloc[0]['userId'] - 1 # resta 1 para obtener el index de pandas.
user0=users_predictions.argsort()[usuario_ver]
# ver los 10 recomendados con mayor puntaje en la predic para este usuario
for i, aRepo in enumerate(user0[-3:]):
    selRepo = df_movies[df_movies['movieId']==(aRepo+1)]
    print(selRepo['title'] ,selRepo['movieId'], 'puntaje:', users_predictions[usuario_ver][aRepo])



usuario_ver = 0 #data.iloc[0]['userId'] - 1 # resta 1 para obtener el index de pandas.
user0=users_predictions.argsort()[usuario_ver]
# ver los 10 recomendados con mayor puntaje en la predic para este usuario
for i, aRepo in enumerate(user0[-3:]):
    selRepo = df_movies[df_movies['movieId']==(aRepo+1)]
    print(selRepo['title'] ,selRepo['movieId'], 'puntaje:', users_predictions[usuario_ver][aRepo])

type(users_predictions)
