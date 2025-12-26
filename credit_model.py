import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_ann_model(input_shape=(5,)):
    """
    Constructs a Deep Learning model for Credit Scoring.
    Includes Dropout layers to mitigate overfitting risks.
    """
    model = Sequential([
        # Initial layer with 12 neurons
        Dense(12, input_dim=input_shape[0], activation='relu'),
        Dropout(0.2),
        
        # Second hidden layer
        Dense(8, activation='relu'),
        
        # Output layer for binary classification (0 or 1)
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Initialize and summarize the architecture
    classifier = build_ann_model()
    classifier.summary()
    print("\n[INFO] Model architecture defined and compiled successfully.")
