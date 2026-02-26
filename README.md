# Progetto Churn Prediction - Revolut/Demo Startup Fintech

## Descrizione
Questo progetto mostra come predire il **churn dei clienti** (abbandono del servizio) per una startup fintech o banca digitale.  
Utilizza un dataset clienti simulato (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) e un modello di **Random Forest** per stimare la probabilità di churn.

L’obiettivo è generare un **report chiaro dei clienti a rischio** e, tramite interfaccia web Streamlit, permettere l’inferenza su nuovi dati.
## Requisiti

- Python 3.9+
- Librerie:
  - pandas
  - scikit-learn
  - streamlit
  - joblib
  - matplotlib (opzionale per grafici)

## Metriche
===============================
Accuracy: 0.7946530147895335
Precision: 0.6540697674418605
Recall: 0.4817987152034261
F1: 0.5548705302096177
ROC AUC: 0.8338223610334766
===========REPORT==============

## Note
Assicurati che i nuovi CSV abbiano le stesse colonne/features usate dal modello.

L’interfaccia e il report mostrano solo le informazioni utili, non tutte le feature del dataset.