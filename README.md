
# House Price Prediction Using Linear Regression

## Overview
This project predicts house prices based on input features like area, number of bedrooms, and bathrooms using a Linear Regression model from the **scikit-learn** library.

## Setup Instructions
1. **Requirements**:  
   Install the following Python libraries:  
   ```bash
   pip install numpy pandas scikit-learn
   ```

2. **Dataset**:  
   - File Name: `House_Price.csv`  
   - Path: `C:\Users\risha\Downloads\OPTIFY\House_Price.csv`  
   - Contains features:  
     - `Area` (in square feet)  
     - `Bedrooms`  
     - `Bathrooms`  
     - `Price` (target variable)  

## Code Workflow
1. **Read the Dataset**: Load the CSV file into a Pandas DataFrame.  
2. **Features and Target**:  
   - Features: `Area`, `Bedrooms`, `Bathrooms`  
   - Target: `Price`  
3. **Model Creation**: Train a `LinearRegression` model on the dataset.  
4. **Prediction Function**:  
   Accepts user inputs for area, bedrooms, and bathrooms to predict the house price.  
5. **Model Evaluation**:  
   Uses test data (`YA` and `Predicted_prices`) to compute the following:  
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  
   - Root Mean Squared Error (RMSE)  

## Code Details

### Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

### Read Dataset
```python
file = r'C:\Users\risha\Downloads\OPTIFY\House_Price.csv'
df = pd.read_csv(file)
print(df)
```

### Model Creation
```python
X = df[['Area ', 'Bedrooms', 'Bathrooms']]
Y = df['Price']
model = LinearRegression()
model.fit(X, Y)
```

### Prediction Function
```python
def predict_price(area, bedroom, bathroom):
    user_input = pd.DataFrame([[area, bedroom, bathroom]], columns=['Area ', 'Bedrooms', 'Bathrooms'])
    Predicted_price = model.predict(user_input)
    return Predicted_price[0]
```

### User Input and Prediction
```python
area = float(input("Enter the area: "))
bedroom = int(input("Enter the number of bedrooms: "))
bathroom = int(input("Enter the number of bathrooms: "))
Predicted_price = predict_price(area, bedroom, bathroom)
print(f"The predicted price for a {area} sqft house with {bedroom} bedroom(s) and {bathroom} bathroom(s): ${Predicted_price:.2f}")
```

### Model Evaluation
```python
YA = [550000, 850000, 782000, 486110, 293000, 705380, 381000, 499000, 497300, 447585]
Predicted_prices = np.array([492341.06101, 838850.6332, 784647.7544, 481467.10021, 298204.82252, 722821.53464, 364332.01102, 481494.92107, 481494.92107, 446244.96628])
mae = mean_absolute_error(YA, Predicted_prices)
mse = mean_squared_error(YA, Predicted_prices)
rmse = np.sqrt(mse)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
```

## Usage
1. Run the script.  
2. Enter inputs for area, bedrooms, and bathrooms.  
3. Get the predicted price.  
4. Evaluate model performance using predefined test data.  

## Outputs
- Predicted house price for given inputs.  
- Evaluation metrics: MAE, MSE, and RMSE.
