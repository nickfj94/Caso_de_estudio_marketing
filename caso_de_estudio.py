"""
Caso de estudio Marketing

Elaborado por: 

Laura Rojas
Cristhian Guzman
Nicolas Florez

"""

## Librerias
import pandas as pd
import numpy as np
import sqlite3 as sql
from matplotlib.pyplot import figure
import seaborn as sns

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



pd.crosstab(index=df_users['rating'], columns=df_users['movieId'], margins=True)
pd.crosstab(index=df_users['rating'], columns=df_users['userId'], margins=True)