import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset and handle missing values
data = pd.read_csv('water_potability.csv')
data.fillna(data.mean(), inplace=True)

# Split the dataset into features and target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Split the dataset into training and test sets
split_ratio = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit the Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

