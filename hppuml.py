# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset (make sure to update the path to your CSV file)
data = pd.read_csv('path_to_your_file.csv')

# Display the first few rows of the dataset
print(data.head())

# Display general info about the dataset (e.g., number of entries, data types)
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Handle missing data
# Fill missing numerical values with the median
data.fillna(data.median(), inplace=True)

# Fill missing categorical values with the mode
data.fillna(data.mode().iloc[0], inplace=True)

# Check again for missing values to ensure they’ve been filled
print(data.isnull().sum())

# Visualize the distribution of the target variable 'SalePrice'
plt.figure(figsize=(8, 6))
sns.histplot(data['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.show()

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Split the data into features (X) and target (y)
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the sizes of the training and test sets
print(f'Training set size: {X_train.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')
