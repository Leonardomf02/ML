### Titanic Dataset - Machine Learning Classification
# Enhanced with insights from the PDF and best practices

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
data = pd.read_csv('train.csv')

# Initial dataset analysis
print("\nDataset Information:")
print(data.info())

print("\nDescriptive Statistics:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

# Handling missing values
imputer = SimpleImputer(strategy='median')
data['Age'] = imputer.fit_transform(data[['Age']])
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(['Cabin'], axis=1, inplace=True)

print("\nMissing Values After Treatment:")
print(data.isnull().sum())

# Encoding categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Feature selection
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = data[features]
y = data['Survived']

# Data normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Function for cross-validation
def evaluate_model(model, X, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    return results.mean(), results.std()

# Models for comparison
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=6),
    'KNN': KNeighborsClassifier(n_neighbors=6),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=24),
    'Logistic Regression': LogisticRegression(max_iter=300, C=0.5, random_state=24)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    cv_mean, cv_std = evaluate_model(model, X, y)
    results[name] = {'Accuracy': acc, 'CV Mean': cv_mean, 'CV Std': cv_std}
    print(f"\n### {name} ###")
    print(f"Accuracy: {acc:.2f}")
    print(f"Cross-Validation Mean: {cv_mean:.2f}, Std: {cv_std:.2f}")
    print(f"Classification Report:\n{classification_report(y_val, y_pred)}")

# Comparison of results
df_results = pd.DataFrame(results).T
print("\nModel Comparison:")
print(df_results)

# Dynamic Conclusion
top_model = df_results['Accuracy'].idxmax()
best_accuracy = df_results.loc[top_model, 'Accuracy']
print("\nConclusion:")
print(f"The best performing model is {top_model} with an accuracy of {best_accuracy:.2f}. This model could be a robust choice for this problem.")

# Feature importance visualization for Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
sns.barplot(x=importances, y=features)
plt.title('Feature Importance - Random Forest')     
plt.show()
