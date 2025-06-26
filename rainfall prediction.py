import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Load the dataset
file_path = r"C:\Users\N Hareesh\Downloads\district wise rainfall normal.csv"
data = pd.read_csv(file_path)

# Show basic information about the dataset
print(data.info())
print(data.head())

# Bar graph for district-wise rainfall
plt.figure(figsize=(15, 8))
plt.bar(data['DISTRICT'], data['ANNUAL'], color='skyblue')
plt.xlabel('District')
plt.ylabel('Annual Rainfall (mm)')
plt.title('District-wise Annual Rainfall')
plt.xticks(rotation=90)  # Rotate the x labels for better readability
plt.tight_layout()
plt.show()

# Prepare data for decision tree
# Assuming 'ANNUAL' is the target variable and other columns are features
X = data.drop(columns=['ANNUAL'])  # Features
y = data['ANNUAL']  # Target variable

# Convert categorical variables to numerical (if any)
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree model (regressor)
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(decision_tree, feature_names=X.columns, filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()
