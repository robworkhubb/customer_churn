import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(path="datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    # Inserisco il csv in un dataframe
    df = pd.read_csv(path)
    # Numero righe e colonne
    #print(df.shape)
    # Conteggio valori
    #print(df["Churn"].value_counts())
    # Percentuale
    #print(df["Churn"].value_counts(normalize=True))
    # Eliminazione colonne inutili
    df.drop('customerID', axis=1, inplace=True)
    # Controllo valori nulli
    #print(df.isnull().sum())
    # Vedo i tipi se è object posso eseguire:
    df["TotalCharges"].dtypes
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Encoding
    # Vedo quante classi posso avere per ogni colonna
    # colonne categoriche
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    #for col in categorical_cols:
        #print(col, df[col].nunique())

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    multiclass_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']


    # Label encoding solo per valori binari
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})
        
    # Target (y)
    y = df['Churn'].map({'Yes':1,'No':0})

    # Feature: le altre colonne
    X = df.drop('Churn', axis=1)
    # One Hot encoding con colonne a multi classe
    
    X = pd.get_dummies(X, columns=multiclass_cols, drop_first=True)
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Ricostruiamo il DataFrame con le colonne corrette
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    return X, y