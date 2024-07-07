# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils import resample

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Data Exploration
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Plot class distribution
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Plot correlation matrix
plt.figure(figsize=(20, 15))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Feature and target split
X = data.drop(columns=['Class'])
y = data['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using oversampling and undersampling
# Separate majority and minority classes
X_majority = X_scaled[y == 0]
X_minority = X_scaled[y == 1]
y_majority = y[y == 0]
y_minority = y[y == 1]

# Upsample minority class
X_minority_upsampled, y_minority_upsampled = resample(X_minority, y_minority,
                                                      replace=True,    # sample with replacement
                                                      n_samples=len(X_majority),  # to match majority class
                                                      random_state=42) # reproducible results

# Combine majority class with upsampled minority class
X_upsampled = np.vstack((X_majority, X_minority_upsampled))
y_upsampled = np.hstack((y_majority, y_minority_upsampled))

# Split the upsampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)

# Model Building
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation
def evaluate_model(y_test, y_pred, model_name):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'{model_name} Evaluation:')
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('-'*50)

    # Plot confusion matrix
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_pred):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()
    plt.show()

evaluate_model(y_test, y_pred_logreg, 'Logistic Regression')
evaluate_model(y_test, y_pred_rf, 'Random Forest')

# Feature Importance for Random Forest
feature_importances = pd.DataFrame(rf.feature_importances_, index=data.columns[:-1], columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

# Visualization of Feature Importances
sns.barplot(x=feature_importances.importance, y=feature_importances.index)
plt.title('Feature Importances')
plt.show()

# Undersample the majority class
X_majority_downsampled, y_majority_downsampled = resample(X_majority, y_majority,
                                                          replace=False,   # sample without replacement
                                                          n_samples=len(X_minority),  # to match minority class
                                                          random_state=42) # reproducible results

# Combine minority class with downsampled majority class
X_downsampled = np.vstack((X_majority_downsampled, X_minority))
y_downsampled = np.hstack((y_majority_downsampled, y_minority))

# Split the downsampled data into training and testing sets
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_downsampled, y_downsampled, test_size=0.2, random_state=42)

# Model Building on downsampled data
# Logistic Regression on downsampled data
logreg_under = LogisticRegression()
logreg_under.fit(X_train_under, y_train_under)
y_pred_logreg_under = logreg_under.predict(X_test_under)

# Random Forest Classifier on downsampled data
rf_under = RandomForestClassifier(n_estimators=100, random_state=42)
rf_under.fit(X_train_under, y_train_under)
y_pred_rf_under = rf_under.predict(X_test_under)

# Evaluation on downsampled data
evaluate_model(y_test_under, y_pred_logreg_under, 'Logistic Regression (Undersampled)')
evaluate_model(y_test_under, y_pred_rf_under, 'Random Forest (Undersampled)')
