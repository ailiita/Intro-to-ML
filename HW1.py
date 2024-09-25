# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:29:08 2024

@author: Sarra Dehili
"""
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

ver = pd.__version__
print(ver)


df = pd.read_csv(r"C:\Users\Sarra Dehili\Documents\ML\1 - intro to ML\laptops.csv")

print(df.head(2))

print("---------------- Laptop rands -----------")
print(df.Brand.nunique())

print("---------------- Number of Nan  -----------")
print(df.isnull().sum())

print("---------------- Max Final Price per Brand  -----------")
print(df.groupby("Brand")[["Final Price"]].max())

med = df["Screen"].median()
print(f"The median value of Screen is : {med}")

mod = df["Screen"].mode(dropna=True)[0]
print(f"The most common value in Screen is : {mod}")


df["Screen"].fillna(mod, inplace=True)
print(df.isnull().sum())
med = df["Screen"].median()
print(f"The median value of Screen is : {med}")


print("---------------- ------- -----------")


# Recuperer colonnes d'interet
newdf = df[['Brand', 'RAM', 'Storage', 'Screen']]
print(newdf)

#Trier par marque et extraire colonne MARQUE Innjoo
group = newdf.groupby('Brand')
print(group)
new_group = group.get_group('Innjoo')
print(new_group)

#obtenir matrice des valeurs
X = new_group[['RAM', 'Storage', 'Screen']].values
print(X)


XTX = np.dot(X.T,X)
print(f'Dot Product of XT and X is :\n{XTX}')

inv_XTX = np.linalg.inv(XTX)
print(f'Inverse matrix of XTX is :\n{inv_XTX}')

y = [1100, 1300, 800, 900, 1000, 1100]

w1 = np.dot(inv_XTX,X.T)
w = np.dot(w1,y)
print(f'dot product gives w= :\n{w}')
print(f' Sum of w give =  :\n{sum(w)}')
