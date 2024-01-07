import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Accuracy values
accuracy_dt = 98
accuracy_svm = 96

# Precision values
precision_dt = 0.98
precision_svm = 0.97

# Recall values
recall_dt = 0.98
recall_svm = 0.96

# F1 score values
f1_score_dt = 0.98
f1_score_svm = 0.96

# Create a dataframe
data = {
    'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Decision Tree': [accuracy_dt, precision_dt, recall_dt, f1_score_dt],
    'SVM': [accuracy_svm, precision_svm, recall_svm, f1_score_svm]
}
df = pd.DataFrame(data)

# Reshape the dataframe for plotting
df_plot = df.melt('Metrics', var_name='Algorithm', value_name='Score')

# Plot the bar graph
plt.figure(figsize=(10, 6))
sns.barplot(x='Metrics', y='Score', hue='Algorithm', data=df_plot, palette='coolwarm')
plt.title('Algorithm Comparison')
plt.ylabel('Score')
plt.ylim(0, 100)
plt.show()
