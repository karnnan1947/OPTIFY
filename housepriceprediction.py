import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
# reading houseprice.csv file from my device
file=r'C:\Users\risha\Downloads\OPTIFY\House_Price.csv'
df=pd.read_csv(file)
print(df)
#features
X=df[['Area ','Bedrooms','Bathrooms']]
#target
Y=df['Price']
#creating model
model=LinearRegression()
model.fit(X,Y)
def predict_price(area,bedroom,bathroom):
    user_input=pd.DataFrame([[area,bedroom,bathroom]],columns=['Area ','Bedrooms','Bathrooms'])
    Predicted_price=model.predict(user_input)
    return Predicted_price[0]
area=float(input(" enter the area:"))
bedroom=int(input("enter the no:bedroom "))
bathroom=int(input("enter the no:bathroom :"))
Predicted_price=predict_price(area,bedroom,bathroom)
print(f"The predicted price for a {area} sqft house with {bedroom} bedroom and {bathroom} bathroom : ${Predicted_price}")
# evaluating the models performance
"""
YA = [550000, 850000, 782000, 486110, 293000, 705380, 381000, 499000, 497300, 447585]
Predicted_prices = np.array([492341.06101, 838850.6332, 784647.7544, 481467.10021, 298204.82252, 722821.53464, 364332.01102, 481494.92107, 481494.92107, 446244.96628])
mae=mean_absolute_error(YA,Predicted_prices)
mse=mean_squared_error(YA,Predicted_prices)
rmse=np.sqrt(mse)
print(f"Mean Absolute Error(MAE): {mae:.2f}")
print(f"Mean Square Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
"""