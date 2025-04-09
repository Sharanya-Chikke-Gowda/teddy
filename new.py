import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

data = pd.read_csv('Credit Default.csv')
# print (data.describe ()) 
# print (data)

data.dropna (inplace = True)
# print (data.dropna)
x = data.drop(['Default'], axis = 1) #1- column & 0-row
y = data['Default']
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size = 0.2)
Scaler = StandardScaler()
x_train_Scaled = Scaler.fit_transform(x_train)
x_test_Scaled = Scaler.transform(x_test)
model = LogisticRegression()
model.fit (x_train_Scaled, y_train)
y_pred = model.predict (x_test_Scaled)
d = pd.read_csv ("abcd.csv")
d_Scaled = Scaler.fit_transform (d)
y_pred_final = model.predict (d_Scaled)
print (y_pred_final)

for i in y_pred_final:
    if i == 0:
        print ("Loan Approved")
    else:
        print ("Loan Denied")

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")