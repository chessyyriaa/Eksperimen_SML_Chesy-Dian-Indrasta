import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# --- KONFIGURASI PARAMETER ---
best_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5
}

def load_data(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {full_path}")
        
    return pd.read_csv(full_path)

def main():

    mlflow.set_tracking_uri("file:./mlruns") 

    mlflow.set_experiment("Churn_Prediction_Local")


    mlflow.autolog()

    print("Loading Data...")
    df = load_data("churn_data_clean.csv")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Model...")
    
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42, **best_params)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Final Accuracy: {acc}")
        print("✅ Model berhasil disimpan di folder lokal ./mlruns ")

if __name__ == "__main__":
    main()
