from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target = 'label'
X = data[features].values
y = data[target].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Initialize the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Predict on the testing set
y_pred_decision_tree = decision_tree.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Calculate accuracy
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("Accuracy - Decision Tree: {:.2f}%".format(accuracy_decision_tree * 100))
print("Accuracy - SVM: {:.2f}%".format(accuracy_svm * 100))

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score for Decision Tree
precision_decision_tree = precision_score(y_test, y_pred_decision_tree, average='weighted')
recall_decision_tree = recall_score(y_test, y_pred_decision_tree, average='weighted')
f1_score_decision_tree = f1_score(y_test, y_pred_decision_tree, average='weighted')

print("Precision - Decision Tree: {:.2f}".format(precision_decision_tree))
print("Recall - Decision Tree: {:.2f}".format(recall_decision_tree))
print("F1 Score - Decision Tree: {:.2f}".format(f1_score_decision_tree))

# Calculate precision, recall, and F1 score for SVM
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_score_svm = f1_score(y_test, y_pred_svm, average='weighted')

print("Precision - SVM: {:.2f}".format(precision_svm))
print("Recall - SVM: {:.2f}".format(recall_svm))
print("F1 Score - SVM: {:.2f}".format(f1_score_svm))
