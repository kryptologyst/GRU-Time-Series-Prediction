"""Data generation and preprocessing utilities for GRU time series prediction."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from config import config


class TimeSeriesGenerator:
    """Generate synthetic time series data for training and testing."""
    
    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed or config.data.random_seed
        np.random.seed(self.random_seed)
    
    def generate_sine_wave(self, 
                          time_steps: int = None,
                          frequency: float = 1.0,
                          amplitude: float = 1.0,
                          phase: float = 0.0,
                          noise_level: float = None) -> np.ndarray:
        """Generate sine wave with optional noise."""
        time_steps = time_steps or config.data.time_steps
        noise_level = noise_level or config.data.noise_level
        
        t = np.linspace(0, 4 * np.pi * frequency, time_steps)
        signal = amplitude * np.sin(t + phase)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, time_steps)
            signal += noise
        
        return signal
    
    def generate_cosine_wave(self,
                           time_steps: int = None,
                           frequency: float = 1.0,
                           amplitude: float = 1.0,
                           phase: float = 0.0,
                           noise_level: float = None) -> np.ndarray:
        """Generate cosine wave with optional noise."""
        time_steps = time_steps or config.data.time_steps
        noise_level = noise_level or config.data.noise_level
        
        t = np.linspace(0, 4 * np.pi * frequency, time_steps)
        signal = amplitude * np.cos(t + phase)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, time_steps)
            signal += noise
        
        return signal
    
    def generate_trend_with_seasonality(self,
                                      time_steps: int = None,
                                      trend_slope: float = 0.01,
                                      seasonal_amplitude: float = 0.5,
                                      seasonal_period: int = 50,
                                      noise_level: float = None) -> np.ndarray:
        """Generate time series with trend and seasonality."""
        time_steps = time_steps or config.data.time_steps
        noise_level = noise_level or config.data.noise_level
        
        t = np.arange(time_steps)
        
        # Trend component
        trend = trend_slope * t
        
        # Seasonal component
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
        
        # Combine components
        signal = trend + seasonal
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, time_steps)
            signal += noise
        
        return signal
    
    def generate_stock_like_data(self,
                               time_steps: int = None,
                               initial_price: float = 100.0,
                               volatility: float = 0.02,
                               drift: float = 0.001) -> np.ndarray:
        """Generate stock-like data using geometric Brownian motion."""
        time_steps = time_steps or config.data.time_steps
        
        dt = 1.0
        prices = [initial_price]
        
        for _ in range(time_steps - 1):
            random_shock = np.random.normal(0, 1)
            price_change = prices[-1] * (drift * dt + volatility * random_shock * np.sqrt(dt))
            new_price = prices[-1] + price_change
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        return np.array(prices)
    
    def generate_multiple_series(self, 
                               series_types: Dict[str, Dict[str, Any]],
                               time_steps: int = None) -> Dict[str, np.ndarray]:
        """Generate multiple time series with different characteristics."""
        time_steps = time_steps or config.data.time_steps
        series_data = {}
        
        for name, params in series_types.items():
            series_type = params.get('type', 'sine')
            
            if series_type == 'sine':
                series_data[name] = self.generate_sine_wave(time_steps=time_steps, **params.get('params', {}))
            elif series_type == 'cosine':
                series_data[name] = self.generate_cosine_wave(time_steps=time_steps, **params.get('params', {}))
            elif series_type == 'trend_seasonal':
                series_data[name] = self.generate_trend_with_seasonality(time_steps=time_steps, **params.get('params', {}))
            elif series_type == 'stock':
                series_data[name] = self.generate_stock_like_data(time_steps=time_steps, **params.get('params', {}))
            else:
                logger.warning(f"Unknown series type: {series_type}")
        
        return series_data


class DataPreprocessor:
    """Preprocess time series data for GRU training."""
    
    def __init__(self, scaler_type: str = 'minmax'):
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        scaled_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        
        return scaled_data.flatten() if scaled_data.shape[1] == 1 else scaled_data
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming data")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        scaled_data = self.scaler.transform(data)
        return scaled_data.flatten() if scaled_data.shape[1] == 1 else scaled_data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming data")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        original_data = self.scaler.inverse_transform(data)
        return original_data.flatten() if original_data.shape[1] == 1 else original_data
    
    def create_sequences(self, 
                        data: np.ndarray, 
                        window_size: int = None,
                        target_offset: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU input."""
        window_size = window_size or config.model.window_size
        
        X, y = [], []
        for i in range(len(data) - window_size - target_offset + 1):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size + target_offset - 1])
        
        return np.array(X), np.array(y)
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray, 
                   train_split: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        train_split = train_split or config.model.train_split
        
        split_idx = int(len(X) * train_split)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def prepare_for_gru(self, X: np.ndarray) -> np.ndarray:
        """Reshape data for GRU input (samples, time_steps, features)."""
        if X.ndim == 2:
            return X.reshape((X.shape[0], X.shape[1], 1))
        return X


class DataVisualizer:
    """Visualize time series data and results."""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_time_series(self, 
                        data: Dict[str, np.ndarray], 
                        title: str = "Time Series Data",
                        figsize: Tuple[int, int] = (12, 8)):
        """Plot multiple time series."""
        fig, axes = plt.subplots(len(data), 1, figsize=figsize, sharex=True)
        if len(data) == 1:
            axes = [axes]
        
        for i, (name, series) in enumerate(data.items()):
            axes[i].plot(series, label=name, linewidth=1.5)
            axes[i].set_title(f"{name} Time Series")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        axes[-1].set_xlabel("Time Step")
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_predictions(self, 
                        actual: np.ndarray, 
                        predicted: np.ndarray,
                        title: str = "Predictions vs Actual",
                        figsize: Tuple[int, int] = (12, 6)):
        """Plot predictions against actual values."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Time series plot
        time_steps = range(len(actual))
        ax1.plot(time_steps, actual, label='Actual', linewidth=2, alpha=0.8)
        ax1.plot(time_steps, predicted, label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
        ax1.set_title(title)
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(actual, predicted, alpha=0.6, s=20)
        ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax2.set_xlabel("Actual Values")
        ax2.set_ylabel("Predicted Values")
        ax2.set_title("Actual vs Predicted Scatter Plot")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, 
                            history: Dict[str, list],
                            figsize: Tuple[int, int] = (10, 4)):
        """Plot training history."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        epochs = range(1, len(history['loss']) + 1)
        ax.plot(epochs, history['loss'], label='Training Loss', linewidth=2)
        
        if 'val_loss' in history and history['val_loss']:
            ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
        
        ax.set_title('Model Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Default data generator and preprocessor instances
data_generator = TimeSeriesGenerator()
data_preprocessor = DataPreprocessor()
data_visualizer = DataVisualizer()
