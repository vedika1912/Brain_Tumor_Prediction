import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv("training_data.csv")  # Replace "training_data.csv" with the actual file name
target_data = pd.read_csv("training_data_targets.csv", header=None)  # Replace "training_data_target.csv" with the actual file name

# Merge features and target variable
merged_data = pd.concat([data, target_data], axis=1)
merged_data.columns = list(data.columns) + ["Target"]  # Replace "Target" with an appropriate name

# Drop "Primary_Diagnosis" from the entire dataset
merged_data.drop("Primary_Diagnosis", axis=1, inplace=True)

# Separate features and target variable
y = merged_data["Target"]
X = merged_data.drop("Target", axis=1)

# Identify categorical columns
categorical_cols = ["Gender", "Race", "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC", "MUC16", "PIK3CA", "NF1", "PIK3R1", "FUBP1", "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4", "GRIN2A", "IDH2", "FAT4", "PDGFRA"]

# Perform One-Hot Encoding for categorical columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)

# Drop the original categorical columns after encoding
X.drop(categorical_cols, axis=1, inplace=True)

# Concatenate encoded features with the original dataset
X = pd.concat([X, X_encoded], axis=1)

# Convert "Age_at_diagnosis" to numeric, extracting only the first two characters
X["Age_at_diagnosis"] = X["Age_at_diagnosis"].str.extract('(\d{2})').astype(float)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Implement Decision Tree algorithm
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

# Logistic Regression
lr_classifier = LogisticRegression(random_state=42)
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)

# Support Vector Machine
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# Multinomial Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the models
classifiers = {
    'Decision Tree': y_pred_dt,
    'Logistic Regression': y_pred_lr,
    'Support Vector Machine': y_pred_svm,
    'Multinomial Naive Bayes': y_pred_nb,
    'Random Forest Classifier': y_pred_rf
}

for clf_name, y_pred in classifiers.items():
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"\n{clf_name} - Accuracy: {accuracy}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_rep)
'''
# Visualize the data
sns.pairplot(X.join(y), hue="Target", diag_kind="kde")
plt.show()

# Perform EDA
correlation_matrix = X.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
'''
