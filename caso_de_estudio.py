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

## Carga Base de datos

conn= sql.connect("db_movies")

df_movies=pd.read_sql("select * from movies", conn)

df_users = pd.read_sql("select *,datetime(timestamp,'unixepoch') as date from ratings", conn)

df_movies
df_users

### Revision de los datos

df_users['userId'].unique().shape[0]
df_movies['movieId'].unique().shape[0]

### Eliminar la columna  timstamp

df_users.drop(['timestamp'], axis=1,inplace=True)

### union