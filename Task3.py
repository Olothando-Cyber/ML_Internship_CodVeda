import pandas as pd
import sklearn
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("L:\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv")
data.head()
data.isnull().sum()
data.duplicated().sum()
#no duplicates and missing data
data.describe()
data.groupby("State").describe()
data["International plan"] = data["International plan"].map({'Yes':1,'No':0})
data["Voice mail plan"] = data["Voice mail plan"].map({'Yes':1,'No':0})
encoder = OneHotEncoder(sparse_output=False)
encodedData= pd.DataFrame(encoder.fit_transform(data[["State"]]),columns=encoder.get_feature_names_out(["State"]))
data = pd.concat([data,encodedData],axis=1)
data = data.drop(columns=['State'])
numericalData = ["Account length","Number vmail messages","Total day minutes","Total day calls","Total day charge","Total eve minutes","Total eve calls","Total eve charge","Total night minutes"	,"Total night calls"	,"Total night charge","Total intl minutes","Total intl calls","Total intl charge","Customer service calls"]
data[numericalData] = RobustScaler().fit_transform(data[numericalData])

x = data.drop(columns=["Churn"])
y = data["Churn"].astype(int)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
