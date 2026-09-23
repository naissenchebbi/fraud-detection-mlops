import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

print("chargement des données")
DATA_PATH = os.getenv("FRAUD_DATA_PATH", "creditcard.csv")
df = pd.read_csv(DATA_PATH)
print(f"dataset chargé:{len(df)} transactions")
print(F"Fraudes: {df['Class'].sum()} (df{df['Class'].sum()/len(df)*100:.3f}%)")


X=df.drop('Class',axis=1)
y=df['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print("\n⚖️ Application de SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"Après SMOTE : {len(X_train_res)} échantillons")

print("\n🤖 Entraînement de XGBoost...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_res, y_train_res)

y_pred=model.predict(X_test_scaled)
y_proba=model.predict_proba(X_test_scaled)[:,1]


print("\n📊 RÉSULTATS :")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler,'scaler.pkl')

print("\n✅ Modèle et scaler sauvegardés avec succès.")