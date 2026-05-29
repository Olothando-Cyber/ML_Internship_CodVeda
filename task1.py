import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
#read in csv file
data= pd.read_csv("L:/Data Set For Task/1) iris.csv")
#check structure of csv file
data.head()
#check to see if we have any missing data that will cause us
#to need to fill in the missing values
data.isnull().sum()
#remove duplicates so data doesnt have unnecessary data 
data = data.drop_duplicates()
data
#Not a clear indicator as there are other species
data.describe()
#Using decribe based on the species is more accurate
data.groupby("species").describe().T
#Standardizing the data since we no need to be strict between 0 and 1
scaler = StandardScaler()
numerical_data = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
data[numerical_data]= scaler.fit_transform(data[numerical_data])
#Data Encoding using OneHotEncoder since data is nominal 
encoder = OneHotEncoder(sparse_output=False)
data = data.reset_index(drop=True)
encoded_data = pd.DataFrame(encoder.fit_transform(data[["species"]]),columns=encoder.get_feature_names_out(['species']))
encoded_data
data= pd.concat([data,encoded_data],axis=1)
data

#Model Training
x = data.drop(columns=['species','petal_length'])
y = data['petal_length']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
print(data)
