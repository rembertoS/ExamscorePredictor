import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Generate a dataset


data = {
    "Hours_studied": np.random.randint(1 ,10 ,100),
    "Attendace":  np.random.randint(60 , 100, 100),
    "Extracurricular": np.random.randint(0,2,100),
    "Exam_Score": None,
}


data = pd.DateFrame(data)
data["Exam_Score"] = (
    5 * data["Hours_Studied"] + 0.5 * data["Extracurricular"] + np.random.normal(0, 5, 100)
)


# Explore the Dataset

print("\nFirst 5 rows of the dataset: ")
print(data.head())
print("\nDataset statistics: ")
print(data.describe())

#Visualize the Data

sns.pairplot(data, diag_kind="kde")
plt.show()

# Correlation Heatmap 

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Split the data into training and Testing sets

x = data.drop(columns = ["Exam_Score"])
y = data["Exam_Score"]
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

# Train the model 

model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate the model 
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test , y_pred)
r2 = r2_score(y_test , y_pred)

print("\nModel Performance: ")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Visualize Predictions 

plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min() , y_test.max()], '--', color='red')
plt.title("Actual vs Predicted Exam Scores")
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.show()

# Analyze the Coefficients
coefficients = pd.DataFrame(model.coef_, x.columns , columns=["Coefficient"])
print("\nFeature Coefficients: ")
print(coefficients)
