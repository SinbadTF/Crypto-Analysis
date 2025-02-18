import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class MLPrediction:
    def __init__(self, df, prediction_window=24, sequence_length=10):
        self.df = df
        self.prediction_window = prediction_window
        self.sequence_length = sequence_length
        self.prices = df['close'].values
        self.scaler = MinMaxScaler()
        
    def _prepare_data(self):
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.prices.reshape(-1, 1))
        
        # Create sequences
        X = []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
        X = np.array(X)
        
        return X
    
    def _create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(self.sequence_length, 1)),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def predict(self):
        # Prepare data
        X = self._prepare_data()
        
        # Create and train model
        model = self._create_model()
        model.fit(X, self.scaler.transform(self.prices[self.sequence_length:].reshape(-1, 1)),
                 epochs=50, batch_size=32, verbose=0)
        
        # Generate predictions
        last_sequence = self.scaler.transform(self.prices[-self.sequence_length:].reshape(-1, 1))
        predictions = []
        
        current_sequence = last_sequence.reshape((1, self.sequence_length, 1))
        for _ in range(self.prediction_window):
            next_pred = model.predict(current_sequence, verbose=0)
            predictions.append(float(self.scaler.inverse_transform(next_pred)[0, 0]))
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        return predictions