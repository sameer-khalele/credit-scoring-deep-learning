import pandas as pd
from data_generator import generate_synthetic_data
from credit_model import build_ann_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_training_pipeline():
    # 1. Generate synthetic financial data
    print("[1/4] Generating financial records...")
    raw_data = generate_synthetic_data(samples=2000)
    
    # 2. Preprocessing
    X = raw_data.drop('Default', axis=1)
    y = raw_data['Default']
    
    # Scaling features for better Neural Network performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 3. Initialize the Model
    print("[2/4] Initializing Deep Learning architecture...")
    model = build_ann_model(input_shape=(X_train.shape[1],))
    
    # 4. Training
    print("[3/4] Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    print("[4/4] Training complete. Model is ready for deployment.")
    return model, history

if __name__ == "__main__":
    trained_model, training_history = run_training_pipeline()
