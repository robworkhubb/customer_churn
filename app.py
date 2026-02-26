import streamlit as st
import pandas as pd
import joblib
from data import load_data

model = joblib.load('saves/churn_model.pkl')

X, y = load_data("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")  # oppure CSV nuovo

st.title("Predizione Churn Clienti")
uploaded_file = st.file_uploader("Carica un CSV con nuovi clienti", type="csv")
if uploaded_file:
    # Preprocessing del CSV caricato
    X_new, _ = load_data(uploaded_file)
    
    # Inferenza
    y_prob = model.predict_proba(X_new)[:,1]
    y_pred = model.predict(X_new)
    
    # Report finale
    df_report = pd.DataFrame({
        "CustomerID (index)": X_new.index,
        "Prob_Churn": y_prob,
        "Rischio": pd.cut(y_prob, bins=[0,0.3,0.6,1], labels=['Basso','Medio','Alto'])
    })

    # Ordinare per probabilità di churn
    df_report = df_report.sort_values(by='Prob_Churn', ascending=False)

    # Salva CSV
    df_report.to_csv("reports/churn_report.csv", index=False)

    # Mostra le prime 20 righe
    print(df_report.head(20))
    
    st.dataframe(df_report.head(20))

    # Scarica CSV con le predizioni
    df_report.to_csv("report_predizioni.csv", index=False)
    st.success("Report salvato come report_predizioni.csv")