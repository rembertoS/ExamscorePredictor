import numpy as nd
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

