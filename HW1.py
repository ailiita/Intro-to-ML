import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)


print("---------------- Question 1  Panda version  -----------")
ver = pd.__version__
print(ver)

print("---------------- Question 2 records count  -----------")
df = pd.read_csv(r"C:\Users\admin\Documents\ML\1 - intro to ML\laptops.csv")
print(df.shape[0])
print(df.head(2))

print("---------------- Question 3  Laptop brands -----------")
print(df.Brand.nunique())

print("---------------- Question 4 Missing Values  -----------")
print(df.isnull().sum())

print("---------------- Question 5 Max Final Price per Brand  -----------")
print(df.groupby("Brand")[["Final Price"]].max())

print("---------------- Question 6 Median Value of screen  -----------")
med = df["Screen"].median()
print(f"The median value of Screen is : {med}")

mod = df["Screen"].mode(dropna=True)[0]
print(f"The most common value in Screen is : {mod}")


df["Screen"].fillna(mod, inplace=True)
print(df.isnull().sum())
med = df["Screen"].median()
print(f"The median value of Screen is : {med}")


print("---------------- Question 7 Sum of weights  -----------")

newdf = df[['Brand', 'RAM', 'Storage', 'Screen']]
#print(newdf)
group = newdf.groupby('Brand')
#print(group)
new_group = group.get_group('Innjoo')
print(new_group)

# Get numpy array
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
