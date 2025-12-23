import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os


best_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5
}

def load_data(filename):
    # Mencari file csv di folder yang sama dengan script ini
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {full_path}")
        
    return pd.read_csv(full_path)

def main():
    # 1. Setup MLflow Local
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set nama eksperimen (bebas)
    experiment_name = "Final_Model_Churn_Local"
    mlflow.set_experiment(experiment_name)
    
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")

    # 2. Load & Split Data
    print("Loading data...")
    df = load_data("churn_data_clean.csv")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Training Final Model
    print("Training Final Model...")
    
   
    
    with mlflow.start_run(run_name="Final_Model_Train_Local"):
        # Log parameter manual (sesuai best_params)
        mlflow.log_params(best_params)

        # Training
        model = RandomForestClassifier(random_state=42, **best_params)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Final Accuracy: {acc}")

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        
        # Log Model (Menyimpan file model.pkl)
        mlflow.sklearn.log_model(model, "model")
        
        print("✅ Model berhasil disimpan secara LOKAL di folder ./mlruns")

if __name__ == "__main__":
    main()