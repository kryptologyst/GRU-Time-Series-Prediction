"""Configuration management for GRU Network Implementation."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
from pathlib import Path


class ModelConfig(BaseModel):
    """Model configuration parameters."""
    
    # GRU Architecture
    gru_units: int = Field(default=64, description="Number of GRU units")
    dropout_rate: float = Field(default=0.2, description="Dropout rate for regularization")
    recurrent_dropout: float = Field(default=0.2, description="Recurrent dropout rate")
    
    # Training Parameters
    epochs: int = Field(default=100, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Training batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate for optimizer")
    validation_split: float = Field(default=0.2, description="Validation split ratio")
    
    # Data Parameters
    window_size: int = Field(default=20, description="Sequence window size")
    train_split: float = Field(default=0.8, description="Training data split ratio")
    
    # Early Stopping
    patience: int = Field(default=15, description="Early stopping patience")
    min_delta: float = Field(default=0.001, description="Minimum change for early stopping")


class DataConfig(BaseModel):
    """Data configuration parameters."""
    
    # Data Generation
    time_steps: int = Field(default=1000, description="Number of time steps to generate")
    noise_level: float = Field(default=0.1, description="Noise level for synthetic data")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    
    # Data Sources
    use_synthetic: bool = Field(default=True, description="Use synthetic data")
    data_file: Optional[str] = Field(default=None, description="Path to external data file")


class AppConfig(BaseModel):
    """Application configuration."""
    
    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent)
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "models")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "data")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "logs")
    
    # Database
    database_url: str = Field(default="sqlite:///gru_experiments.db")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")
    experiment_name: str = Field(default="GRU_Time_Series", description="MLflow experiment name")
    
    # Model Configuration
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.data_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)


# Global configuration instance
config = AppConfig()
