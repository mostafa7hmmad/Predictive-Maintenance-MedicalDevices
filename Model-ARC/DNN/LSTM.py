from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_lstm_model(seq_len=14, num_classes=3):
    model = Sequential([
        LSTM(64, input_shape=(seq_len, 1), return_sequences=True),
        BatchNormalization(),
        Dropout(0.5),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
