import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier


def classification_pipeline(classifier, X_train, X_test, y_train, y_test):
    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Print results
    print(f"\n{type(classifier).__name__} - Accuracy: {accuracy}")
    print(f"\n{type(classifier).__name__} - F_score: {f_score}")
    print(f"\n{type(classifier).__name__} - Precision: {precision}")
    print(f"\n{type(classifier).__name__} - Recall: {recall}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_rep)

    # Feature importance for decision tree and random forest
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_
        print("Feature Importances:\n", feature_importances)

# Load the data
data = pd.read_csv("training_data.csv")  
target_data = pd.read_csv("training_data_targets.csv", header=None)  
testdata=pd.read_csv("test_data.csv")

# Encoding target variables
le = LabelEncoder()
encoded_target_data = pd.DataFrame(le.fit_transform(target_data))

# Merge features and target variable
merged_data = pd.concat([data, encoded_target_data], axis=1)
merged_data.columns = list(data.columns) + ["Target"]  

# Drop "Primary_Diagnosis" from the entire dataset
merged_data.drop("Primary_Diagnosis", axis=1, inplace=True)
testdata.drop("Primary_Diagnosis", axis=1, inplace=True)
# Separate features and target variable
y = merged_data["Target"]
X = merged_data.drop("Target", axis=1)
X = X.replace(['not reported', '--'], [np.nan, np.nan])  # Replace 'not reported' and '--' with NaN

# Print the number of missing values in each feature
print("Number of missing values before handling:")
print(X.isnull().sum())

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed=pd.DataFrame(imputer.fit_transform(testdata), columns=testdata.columns)

# Convert "Age_at_diagnosis" to numeric, extracting only the first two characters
X_imputed["Age_at_diagnosis"] = X_imputed["Age_at_diagnosis"].str.extract('(\d{2})').astype(float)
X_test_imputed["Age_at_diagnosis"] = X_test_imputed["Age_at_diagnosis"].str.extract('(\d{2})').astype(float)

# Identify categorical columns
categorical_cols = ["Gender", "Race", "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC", "MUC16", "PIK3CA", "NF1", "PIK3R1", "FUBP1", "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4", "GRIN2A", "IDH2", "FAT4", "PDGFRA"]

# Perform label encoding for categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    X_imputed[col] = label_encoder.fit_transform(X_imputed[col])
    X_test_imputed[col] = label_encoder.fit_transform(X_test_imputed[col])
# Specify feature selection method and parameters
feature_selection_method = 'select_k_best'
num_features = 20

'''
# Create correlation matrix
correlation_matrix = X_imputed.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Print eliminated features based on correlation
threshold = 0.5  # You can adjust this threshold as needed
correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

print("Eliminated Features based on Correlation:")
print(correlated_features)

# Drop correlated features
X_selected = X_imputed.drop(correlated_features, axis=1)

'''

# Perform feature selection
if feature_selection_method == 'select_k_best':
    selector = SelectKBest(f_classif, k=num_features)
elif feature_selection_method == 'rfe':
    selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=num_features, step=1)

X_selected = selector.fit_transform(X_imputed, y)
X_test_selected = selector.transform(X_test_imputed)

# Print selected features
selected_features = X_imputed.columns[selector.get_support()]
print("\nSelected Features:")
print(selected_features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Specify the classification algorithm and hyperparameter grid
clf_input = 'SVC'

if clf_input == 'DecisionTree':
    clf_param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    clf = DecisionTreeClassifier(random_state=42)

elif clf_input == 'LogisticRegression':
    clf_param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
    clf = LogisticRegression(random_state=42)

elif clf_input == 'SVC':
    clf_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly','rbf', 'sigmoid'],
    }
    clf = SVC(random_state=42)

elif clf_input == 'MultinomialNB':
    clf_param_grid = {
        'alpha': [0.1, 0.5, 1.0],
        'fit_prior': [True, False]
    }
    clf = MultinomialNB()

elif clf_input == 'RandomForest':
    clf_param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }
    clf = RandomForestClassifier(random_state=42)

elif clf_input == 'AdaBoost':
    clf_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.001, 0.01, 0.1, 1]
    }
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42), random_state=42)

elif clf_input == 'KNN':
    clf_param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    clf = KNeighborsClassifier()

else:
    print("Invalid classifier choice. Please choose from (DecisionTree, LogisticRegression, SVC, MultinomialNB, RandomForest, KNN).")
    exit()

# Create grid search classifier with precision as the scoring metric
grid_search_classifier = GridSearchCV(clf, clf_param_grid, cv=5,scoring='f1')

# Run grid search for the selected classifier
print(f"\nHyperparameter Tuning for {clf_input}:")
grid_search_classifier.fit(X_train, y_train)

# Print best parameters and corresponding accuracy
print("Best Parameters:", grid_search_classifier.best_params_)
print("Best F1_score:", grid_search_classifier.best_score_)
print('--------------------------------------------------')

# Get the best classifier
best_classifier = grid_search_classifier.best_estimator_

# Evaluate the best classifier on the test set
#After analysis SVM was chosen to be the best model for the given dataset with feature selection
#as k_best (number of features=20)
print("Evaluation on Test data\n")
classification_pipeline(best_classifier, X_train, X_test, y_train, y_test)

predicted_labels = best_classifier.predict(X_test_selected)
predicted_labels = pd.DataFrame(le.inverse_transform(predicted_labels))
predicted_labels.to_csv("predicted_labels.txt", sep='\t', index=False,header=None)
