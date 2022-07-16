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
df.groupby(['month'])['rating'].count().sort_values(ascending=False)
# Cantidad de calificaciones por año
df.groupby(['year_rating'])['rating'].count().sort_values(ascending=False)
# Cantidad de calificaciones por año de la pelicula
df.groupby(['year'])['rating'].count().sort_values(ascending=False)

### ALGORITMO

