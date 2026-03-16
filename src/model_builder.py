import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from .config import LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT_RATE, LEARNING_RATE

def build_dual_lstm_model(time_steps, num_features):
    """
    Membangun arsitektur Jaringan Saraf Tiruan LSTM Dua Lapis (Dual-Layer)
    """
    inputs = Input(shape=(time_steps, num_features))
    
    # Layer 1
    x = LSTM(LSTM_UNITS_1, activation="tanh", return_sequences=True)(inputs)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Layer 2
    x = LSTM(LSTM_UNITS_2, activation="tanh")(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Output Layer
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mse")
    
    return model

def get_callbacks():
    """Mengonfigurasi callbacks untuk pelatihan cerdas"""
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    return [lr_scheduler, early_stop]
