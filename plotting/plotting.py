import seaborn as sns

#Example
metrics = {
    'Rouge-1 F-Measure': 0.2878,
    'Rouge-1 Precision': 0.3117,
    'Rouge-1 Recall': 0.2758,
    'Rouge-2 F-Measure': 0.0398,
    'Rouge-2 Precision': 0.0434,
    'Rouge-2 Recall': 0.0379,
    'Rouge-L F-Measure': 0.1579,
    'Rouge-L Precision': 0.1715,
    'Rouge-L Recall': 0.1510,
    'Rouge-L Sum F-Measure': 0.2381,
    'Rouge-L Sum Precision': 0.2583,
    'Rouge-L Sum Recall': 0.2278
}

names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(10, 5))
sns.barplot(x=values, y=names, palette='viridis')
plt.xlabel('Values')
plt.title('Metrics')
plt.show()