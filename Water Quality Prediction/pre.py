import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('water_potability.csv')

# Handling missing values
data.fillna(method='ffill', inplace=True)

# Split the dataset into features and target
X = data.drop(columns=['Potability'])
y = data['Potability']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally, you can scale or normalize your features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Now, X_train and X_test contain the preprocessed features, and y_train and y_test contain the corresponding labels.
