import pandas as pd
import sklearn
import math
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data= pd.read_csv("L:/Data Set For Task/1) iris.csv")
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

model = LinearRegression()
RegModel= model.fit(x_train,y_train)
print(RegModel.intercept_)
print(RegModel.coef_)
y_pred= RegModel.predict(x_test)
print("MSE", mean_squared_error(y_test, y_pred),"RMSE",math.sqrt(mean_squared_error(y_test,y_pred)))
#The model is quite accurate as our model predictions are only 0.1718 standard deviations away from the true value.
print("R-Squared Score", r2_score(y_test,y_pred))
#The model outputs a high percentage which is 96.9% meaning
#96.0% of the variation of the petal length can be accuartely explained by the feauteres chosen.
