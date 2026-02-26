from sklearn.model_selection import train_test_split
from data import load_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform (X_test)

model = RandomForestClassifier(n_estimators= 100, class_weight="balanced")
model.fit(X_train, y_train)

print("===================================")
# Inferenza su test set
y_pred = model.predict(X_test)       # Classi 0/1
y_prob = model.predict_proba(X_test)[:,1]  # Probabilità di churn
print("Prime 10 predizioni (classi):", y_pred[:10])
print("Prime 10 probabilità di churn:", y_prob[:10])

print("===================================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print('===========REPORT==============')
df_report = pd.DataFrame({
    "Customer (index)": X_test.index,
    "Prob_Churn": y_prob,
    "Rischio": pd.cut(y_prob, bins=[0,0.3,0.6,1], labels=['Basso','Medio','Alto'])
})

# Ordinare per probabilità di churn
df_report = df_report.sort_values(by='Prob_Churn', ascending=False)

# Salva CSV compatto
df_report.to_csv("reports/churn_report_compatto.csv", index=False)

# Mostra le prime 20 righe
print(df_report.head(20))

joblib.dump(model, "saves/churn_model.pkl")