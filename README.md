# SA - 1 - scatter-plot-between-cylinder-vs-Co2Emission

# Program developed by : ADITYAH M S
# Register Number : 212223220002

# PROGRAM :
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------
# Load dataset from local path
# ----------------------------
file_path = r"C:\Users\admin\Downloads\FuelConsumption (1).csv"
data = pd.read_csv(file_path)

print("Dataset Preview:")
print(data.head())
print("\nAvailable Columns:", list(data.columns))
```
```
# ----------------------------
# Q1: Scatter plot - CYLINDERS vs CO2 Emission (green)
# ----------------------------
plt.figure()  # start a new figure
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='green')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.title("Cylinders vs CO2 Emissions")
plt.show()   # this will open the first chart separately
```
```
# ----------------------------
# Q2: Scatter plot - CYLINDERS vs CO2 & ENGINESIZE vs CO2
# ----------------------------
plt.figure()  # new figure
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='blue', label="Cylinders vs CO2")
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='red', label="EngineSize vs CO2")
plt.xlabel("X-axis values")
plt.ylabel("CO2 Emissions")
plt.title("Cylinders vs CO2 & EngineSize vs CO2")
plt.legend()
plt.show()   # second chart appears separately
```
```
# ----------------------------
# Q3: Scatter plot - CYLINDERS vs CO2, ENGINESIZE vs CO2, FUELCONSUMPTION_COMB vs CO2
# ----------------------------
plt.figure()  # new figure
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='blue', label="Cylinders vs CO2")
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='red', label="EngineSize vs CO2")
plt.scatter(data['FUELCONSUMPTION_COMB'], data['CO2EMISSIONS'], color='green', label="FuelConsumption vs CO2")
plt.xlabel("X-axis values")
plt.ylabel("CO2 Emissions")
plt.title("Multiple Comparisons with CO2 Emission")
plt.legend()
plt.show()   # third chart appears separately
```
```
# ----------------------------
# Q4: Model - CYLINDERS vs CO2
# ----------------------------
X = data[['CYLINDERS']]
y = data['CO2EMISSIONS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_cyl = LinearRegression()
model_cyl.fit(X_train, y_train)
y_pred_cyl = model_cyl.predict(X_test)
print("\nModel 1 (Cylinders vs CO2) R2 Score:", r2_score(y_test, y_pred_cyl))
```
```
# ----------------------------
# Q5: Model - FUELCONSUMPTION_COMB vs CO2
# ----------------------------
X2 = data[['FUELCONSUMPTION_COMB']]
y2 = data['CO2EMISSIONS']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model_fuel = LinearRegression()
model_fuel.fit(X2_train, y2_train)
y2_pred = model_fuel.predict(X2_test)
print("Model 2 (FuelConsumption vs CO2) R2 Score:", r2_score(y2_test, y2_pred))
```
```
# ----------------------------
# Q6: Train model with different train-test ratios
# ----------------------------
ratios = [0.1, 0.2, 0.3, 0.4]
print("\nModel Accuracies for Different Train-Test Splits:")

for r in ratios:
    X_train, X_test, y_train, y_test = train_test_split(data[['FUELCONSUMPTION_COMB']],
                                                        data['CO2EMISSIONS'],
                                                        test_size=r, random_state=42)
    temp_model = LinearRegression()
    temp_model.fit(X_train, y_train)
    y_pred = temp_model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"Train-Test Split {1-r:.0%} - {r:.0%} => Accuracy (R2 Score): {score:.4f}")
```
# OUTPUT :

<img width="796" height="679" alt="Screenshot 2025-09-21 105533" src="https://github.com/user-attachments/assets/9a5e93b9-6bf4-460b-9ac0-ebf24d0da6be" />
<img width="793" height="676" alt="Screenshot 2025-09-21 110028" src="https://github.com/user-attachments/assets/3b3108f5-dd86-43f5-b9c1-5999b40d7521" />
<img width="795" height="677" alt="Screenshot 2025-09-21 110048" src="https://github.com/user-attachments/assets/44556cd7-ff36-4bdd-ba88-0b4066e83055" />

# RESULT :
Thus, the program has been executed successfully.
