import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Charger les données
print("📥 Chargement des données...")
df = pd.read_csv('creditcard.csv')
print(f"✅ Dataset chargé : {len(df)} transactions")
print(f"Fraudes : {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# 2. Séparer features et target
X = df.drop('Class', axis=1)
y = df['Class']

# 3. Split AVANT SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. SMOTE sur le train uniquement
print("\n⚖️ Application de SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"Après SMOTE : {len(X_train_res)} échantillons")

# 6. Entraîner Random Forest
print("\n🌲 Entraînement de Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)

# 7. Évaluer
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 RÉSULTATS – RANDOM FOREST :")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# 8. Sauvegarder
joblib.dump(model, 'fraud_model_rf.pkl')
print("\n✅ Modèle Random Forest sauvegardé !")