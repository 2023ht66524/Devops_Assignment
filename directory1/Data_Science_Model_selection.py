#Data Science Model Selection for the given malware data

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Read the given data files
file_path_malware = os.path.join('data', 'train_malware.csv')
file_path_benign = os.path.join('data', 'train_benign.csv')
file_path_test = os.path.join('data', 'test.xlsx')
malware_data = pd.read_csv(file_path_malware)
malware_data['class'] = 1  # Adding a new label 'class' with value 1 for malware
benign_data = pd.read_csv(file_path_benign)
benign_data['class'] = 0  # Adding a new label 'class' with value 0 for benign

xls_file = pd.ExcelFile(file_path_test, engine='openpyxl')
test_data_xls = pd.read_excel(xls_file, sheet_name='Sheet1')
test_data_xls.to_csv('test.csv', index=False)  #reading the direct excel file causing error with data types, so converting to csv
test_data = pd.read_csv('test.csv') 

# Combine malware and benign data into one DataFrame
train_data = pd.concat([malware_data, benign_data])
print("Train data shape:",train_data.shape)
print("Test data shape:",test_data.shape)


# Load train data
X_train = train_data.drop('class', axis=1) # Features
y_train = train_data['class'] # Target variable

# Load test data
X_test = test_data.drop('Class ', axis=1) # Selecting all Features expect Class
y_test = test_data['Class '] # Selecting only Class

# Model Selection & Model Training
# Model 1: Decision Tree
decision_tree = DecisionTreeClassifier() #default values #
decision_tree.fit(X_train, y_train)
decision_tree.get_params()
dt_pred = decision_tree.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)


# Model 2: Random Forest
random_forest = RandomForestClassifier() # n_estimators=100 # default values  # The default n_estimator 100 looks best based on the accuracy, tested details avilable in ids.pynb rough book.  
random_forest.fit(X_train, y_train)
random_forest.get_params()
rf_pred = random_forest.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_classification_report = classification_report(y_test, rf_pred)

# Model 3: Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Model 4: Gradient Boosting Machines (GBM)
gbm = GradientBoostingClassifier()  
gbm.fit(X_train, y_train)
gbm_pred = gbm.predict(X_test)
gbm_accuracy = accuracy_score(y_test, gbm_pred)
gbm_classification_report = classification_report(y_test, gbm_pred)

# Model 5: Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', max_iter=10000) # To resolve ConvergenceWarning: lbfgs failed to converge (status=1) added max_iter
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
log_reg_classification_report = classification_report(y_test, log_reg_pred)

# Print the accuracies of all models & classification reports for RF,GBM and LR
print("Decision Tree Accuracy on Test Set:", dt_accuracy)
print("Random Forest Accuracy on Test Set:", rf_accuracy)
print("Support Vector Machine Accuracy on Test Set:", svm_accuracy)
print("Gradient Boosting Machines Accuracy on Test Set:", gbm_accuracy)
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Random Forest Classification Report:")
print(rf_classification_report)
print("Gradient Boosting Machines Classification Report:")
print(gbm_classification_report)
print("Logistic Regression Classification Report:")
print(log_reg_classification_report)

# Evaluate the best model on the test set
best_model = max(dt_accuracy, rf_accuracy, svm_accuracy, gbm_accuracy, log_reg_accuracy )
if best_model == log_reg_accuracy:
 selected_model = log_reg
elif best_model == rf_accuracy:
 selected_model = random_forest
elif best_model == svm_accuracy:
 selected_model = svm
elif best_model == gbm_accuracy:
 selected_model = gbm
else:
 selected_model = decision_tree

print("Best model based on Accuracy is:", selected_model)

