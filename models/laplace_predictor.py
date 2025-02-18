import numpy as np
from scipy import signal
from scipy.fft import fft, ifft

class LaplacePrediction:
    def __init__(self, df, prediction_window=24):
        self.df = df
        self.prediction_window = prediction_window
        self.prices = df['close'].values
        
    def _apply_laplace_transform(self):
        # Apply FFT to get frequency domain representation
        fft_result = fft(self.prices)
        
        # Create frequency domain filter (low-pass filter)
        freq = np.fft.fftfreq(len(self.prices))
        filter_mask = np.abs(freq) < 0.1  # Adjust cutoff frequency as needed
        filtered_fft = fft_result * filter_mask
        
        # Convert back to time domain
        filtered_signal = ifft(filtered_fft)
        return np.real(filtered_signal)
    
    def _extract_trend(self):
        # Apply Savitzky-Golay filter for trend extraction
        window_length = min(15, len(self.prices) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        trend = signal.savgol_filter(self.prices, window_length, 3)
        return trend
    
    def predict(self):
        # Get the filtered signal
        filtered_signal = self._apply_laplace_transform()
        trend = self._extract_trend()
        
        # Combine filtered signal and trend for prediction
        last_value = self.prices[-1]
        trend_slope = (trend[-1] - trend[-2])
        
        # Generate future predictions
        predictions = []
        for i in range(self.prediction_window):
            next_value = last_value + trend_slope * (i + 1)
            # Add some oscillation based on filtered signal pattern
            oscillation = filtered_signal[-1] - filtered_signal[-2]
            next_value += oscillation * np.sin(i * np.pi / 4)
            predictions.append(float(next_value))
        
        return predictions