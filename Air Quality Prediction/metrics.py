import matplotlib.pyplot as plt
import seaborn as sns

# Algorithm metrics
accuracy = [99.95, 99.29, 99.31]
precision = [0.99, 0.99, 0.99]
recall = [0.99, 0.99, 0.99]
f1_score = [0.99, 0.99, 0.99]

# Algorithm labels
algorithms = ['Random Forest', 'K Nearest Neighbors', 'Support Vector Machine']

# Plotting the accuracy with values displayed on top of bars
plt.figure(figsize=(8, 4))
ax = sns.barplot(x=algorithms, y=accuracy)
plt.ylim([0, 100])
plt.title('Algorithm Accuracy')
plt.ylabel('Accuracy (%)')
for i, v in enumerate(accuracy):
    ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
plt.show()

# Plotting precision, recall, and F1 score using a bar graph
metrics = ['Precision', 'Recall', 'F1 Score']
metrics_values = [precision, recall, f1_score]

plt.figure(figsize=(8, 4))
for i, metric in enumerate(metrics):
    ax = sns.barplot(x=algorithms, y=metrics_values[i], label=metric)
    ax.legend()
    plt.ylim([0, 1])
    plt.title('Algorithm Metrics')
    plt.ylabel('Score')
plt.show()

