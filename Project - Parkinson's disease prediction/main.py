# Parkinson's Disease Prediction Project

# Step 1: Install Required Libraries
# Uncomment and run these commands to install libraries
# !pip install ydata-profiling
# !conda install -c conda-forge pandas-profiling
# !pip install visions

# If you encounter profiling errors, use:
# !pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

# Step 2: Import Required Libraries
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, cohen_kappa_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Step 3: Load and Explore the Data
data_path = 'parkinsons.data'  # Replace with your dataset path
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("First 5 Records:")
print(df.head())
print("Column Data Types:")
print(df.dtypes)

# Check for Missing Values
print("Missing Values in Each Column:")
print(df.isna().sum())

# Display Basic Statistics
print(df.describe())

# Step 4: Visualize the Data
# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Histogram for Target Variable
plt.figure(figsize=(8, 5))
df['status'].hist()  # Target variable: 1 (Parkinson's), 0 (Healthy)
plt.xlabel('Status')
plt.ylabel('Frequency')
plt.title('Distribution of Status')
plt.show()

# Barplot for NHR by Status
plt.figure(figsize=(10, 6))
sns.barplot(x="status", y="NHR", data=df)
plt.title('NHR by Status')
plt.show()

# Barplot for HNR by Status
plt.figure(figsize=(10, 6))
sns.barplot(x="status", y="HNR", data=df)
plt.title('HNR by Status')
plt.show()

# Distribution Plots
rows, cols = 3, 7
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 8))
columns = df.columns[1:]
index = 0
for i in range(rows):
    for j in range(cols):
        if index < len(columns):
            sns.histplot(df[columns[index]], ax=ax[i][j], kde=True)
            ax[i][j].set_title(columns[index])
            index += 1
plt.tight_layout()
plt.show()

# Step 5: Prepare Data for Machine Learning
# Drop the 'name' column (not relevant for predictions)
df = df.drop(['name'], axis=1)

# Split Features (X) and Target (Y)
X = df.drop('status', axis=1)
Y = df['status']

# Split Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 6: Build and Evaluate Models
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
log_reg_preds = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(Y_test, log_reg_preds))

# Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, Y_train)
rf_preds = rf_clf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(Y_test, rf_preds))

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, Y_train)
dt_preds = dt_clf.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(Y_test, dt_preds))

# Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)
nb_preds = nb_clf.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(Y_test, nb_preds))

# K-Nearest Neighbors
knn_clf = KNeighborsClassifier(weights='distance')
knn_clf.fit(X_train, Y_train)
knn_preds = knn_clf.predict(X_test)
print("KNN Accuracy:", accuracy_score(Y_test, knn_preds))

# Support Vector Machine (SVM)
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, Y_train)
svm_preds = svm_clf.predict(X_test)
print("SVM Accuracy:", accuracy_score(Y_test, svm_preds))

# Step 7: Save the Best Model (SVM in this case)
pickle.dump(svm_clf, open('parkinsons_model.pkl', 'wb'))
print("SVM Model Saved as 'parkinsons_model.pkl'")

# Step 8: Load the Model and Make Predictions
loaded_model = pickle.load(open('parkinsons_model.pkl', 'rb'))
predictions = loaded_model.predict(X_test)
print("Loaded Model Accuracy:", accuracy_score(Y_test, predictions))

# Display Results
result = pd.DataFrame({"Actual": Y_test, "Predicted": predictions})
print(result.head())
