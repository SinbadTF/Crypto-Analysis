import numpy as np
from .laplace_predictor import LaplacePrediction
from .ml_predictor import MLPrediction

class CombinedPrediction:
    def __init__(self, df, prediction_window=24):
        self.df = df
        self.prediction_window = prediction_window
        self.laplace_predictor = LaplacePrediction(df, prediction_window)
        self.ml_predictor = MLPrediction(df, prediction_window)
    
    def predict(self):
        # Get predictions from both models
        laplace_predictions = self.laplace_predictor.predict()
        ml_predictions = self.ml_predictor.predict()
        
        # Combine predictions with equal weights
        combined_predictions = []
        for i in range(self.prediction_window):
            combined_value = (laplace_predictions[i] + ml_predictions[i]) / 2
            combined_predictions.append(float(combined_value))
        
        return combined_predictions