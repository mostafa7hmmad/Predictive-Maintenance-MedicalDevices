from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from models.optimizers import optimizers

def build_ann_model(input_dim=14, num_classes=3, optimizer='ANN_Adam'):
    if optimizer == 'ANN_SGDM':
        model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
    else:
        model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
    model.compile(
        optimizer=optimizers[optimizer],
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
