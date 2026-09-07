# bee-colony-and-honey-production-analysis
Bee colony and honey production data for analysing trends and predicting honey yield
# Bee Colony and Honey Production Analysis
# Trend Analysis + Honey Yield Prediction

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("bee_colony_honey_production.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------------------------
# 2. Clean Dataset
# --------------------------------------------------

df = df.dropna()

# Display basic statistics
print("\nStatistical Summary:")
print(df.describe())

# --------------------------------------------------
# 3. Trend Analysis
# --------------------------------------------------

# Honey production trend
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["Honey_Production"], marker="o")
plt.xlabel("Year")
plt.ylabel("Honey Production")
plt.title("Honey Production Trend")
plt.grid(True)
plt.show()

# Bee colony trend
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["Bee_Colonies"], marker="o")
plt.xlabel("Year")
plt.ylabel("Number of Bee Colonies")
plt.title("Bee Colony Trend")
plt.grid(True)
plt.show()

# --------------------------------------------------
# 4. Relationship Between Colonies and Honey
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(df["Bee_Colonies"], df["Honey_Production"])
plt.xlabel("Number of Bee Colonies")
plt.ylabel("Honey Production")
plt.title("Bee Colonies vs Honey Production")
plt.grid(True)
plt.show()

# --------------------------------------------------
# 5. Correlation Analysis
# --------------------------------------------------

print("\nCorrelation:")
print(df[["Bee_Colonies", "Honey_Production"]].corr())

# --------------------------------------------------
# 6. Prepare Data for Machine Learning
# --------------------------------------------------

X = df[["Bee_Colonies"]]
y = df["Honey_Production"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 7. Train Linear Regression Model
# --------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# 8. Predict Honey Production
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(result)

# --------------------------------------------------
# 9. Model Evaluation
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)

# --------------------------------------------------
# 10. Regression Line
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(X_test, y_test, label="Actual")

plt.plot(
    X_test,
    y_pred,
    label="Predicted"
)

plt.xlabel("Number of Bee Colonies")
plt.ylabel("Honey Production")
plt.title("Honey Production Prediction")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# 11. Predict Honey Production for New Colony Count
# --------------------------------------------------

new_colonies = [[1000]]

prediction = model.predict(new_colonies)

print("\nPredicted Honey Production for 1000 colonies:",
      prediction[0])
