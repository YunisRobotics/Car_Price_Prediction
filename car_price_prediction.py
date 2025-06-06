import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

#loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv('car data.csv')

car_dataset.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)
car_dataset.replace({'Seller_Type':{'Dealer':0, 'Individual':1}}, inplace=True)
car_dataset.replace({'Transmission':{'Manual':0, 'Automatic':1}}, inplace=True)

X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis = 1)
Y = car_dataset['Selling_Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

lin_reg_model = LinearRegression()

lin_reg_model.fit(X_train, Y_train)

training_data_predict_lr = lin_reg_model.predict(X_train)
test_data_predict_lr = lin_reg_model.predict(X_test)

error_score = metrics.r2_score(Y_train, training_data_predict_lr)
print("R squared Error : ", error_score)

test_error_score = metrics.r2_score(Y_test, test_data_predict_lr)
print("Test R squared Error : ", test_error_score)

plt.scatter(Y_test, test_data_predict_lr, color = 'blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Test Data)")
plt.show()


lasso_reg_model = Lasso()

lasso_reg_model.fit(X_train, Y_train)

training_data_predict_lasso = lasso_reg_model.predict(X_train)
test_data_predict_lasso = lasso_reg_model.predict(X_test)

error_score = metrics.r2_score(Y_train, training_data_predict_lasso)
print("R squared Error : ", error_score)


test_error_score = metrics.r2_score(Y_test, test_data_predict_lasso)
print("Test R squared Error : ", test_error_score)

plt.scatter(Y_test, test_data_predict_lasso, color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Test Data)")
plt.show()

def predict_price():
    print("Enter car details to predict the selling price:")
    
    year = int(input("Year (e.g., 2015): "))
    present_price = float(input("Present Price (in dollars, e.g., 5500): "))
    kms_driven = int(input("Kms Driven (e.g., 27000): "))
    
    print("Fuel Type: 0 = Petrol, 1 = Diesel, 2 = CNG")
    fuel_type = int(input("Fuel Type (0/1/2): "))
    
    print("Seller Type: 0 = Dealer, 1 = Individual")
    seller_type = int(input("Seller Type (0/1): "))
    
    print("Transmission: 0 = Manual, 1 = Automatic")
    transmission = int(input("Transmission (0/1): "))
    
    owner = int(input("Number of previous owners (0/1/2/...): "))
    
    # Prepare input dataframe with the same order as training features
    input_data = pd.DataFrame([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]], 
                              columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])
    
    # Predict with Linear Regression
    price_lr = lin_reg_model.predict(input_data)[0]
    print(f"\nPredicted Selling Price (Linear Regression): {price_lr:.2f} dollars")
    
    # Predict with Lasso Regression
    price_lasso = lasso_reg_model.predict(input_data)[0]
    print(f"Predicted Selling Price (Lasso Regression): {price_lasso:.2f} dollars")

# Call the function
predict_price()
