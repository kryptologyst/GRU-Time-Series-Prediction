# Advanced GRU Time Series Prediction

A modern, production-ready implementation of Gated Recurrent Unit (GRU) networks for time series prediction with comprehensive experiment tracking, interactive UI, and multiple architecture options.

## Features

- **Multiple GRU Architectures**: Standard, Stacked, Bidirectional, and Attention-based models
- **Interactive Web UI**: Streamlit-based interface for model training and visualization
- **Experiment Tracking**: MLflow integration with SQLite database for comprehensive experiment management
- **Advanced Data Generation**: Multiple synthetic time series patterns (sine, cosine, trend+seasonality, stock-like)
- **Modern ML Stack**: TensorFlow 2.x, Pydantic configuration, structured logging
- **Production Ready**: Error handling, model versioning, automated callbacks

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 0077_GRU_network_implementation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface
```bash
python main.py
```

### Interactive Web UI
```bash
streamlit run streamlit_app.py
```

## Project Structure

```
0077_GRU_network_implementation/
├── config.py              # Configuration management with Pydantic
├── database.py             # SQLAlchemy models and database management
├── data_generator.py       # Time series data generation and preprocessing
├── gru_model.py           # Advanced GRU model implementations
├── main.py                # Command line interface
├── streamlit_app.py       # Interactive web application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/                  # Generated data and visualizations
├── models/                # Saved model files
├── logs/                  # Application logs
└── mlruns/               # MLflow experiment tracking
```

## Model Architectures

### 1. Standard GRU
- Single GRU layer with batch normalization
- Dropout regularization
- Dense output layer

### 2. Stacked GRU
- Multiple GRU layers for complex pattern learning
- Progressive unit reduction
- Enhanced capacity for long sequences

### 3. Bidirectional GRU
- Processes sequences in both directions
- Better context understanding
- Improved performance on symmetric patterns

### 4. Attention GRU
- Attention mechanism for long-term dependencies
- Weighted context vectors
- Advanced sequence modeling

## Data Generation Options

- **Sine Wave**: Pure sinusoidal patterns with configurable frequency and noise
- **Cosine Wave**: Cosine-based patterns for phase-shifted signals
- **Trend + Seasonality**: Combined linear trends with seasonal components
- **Stock-like**: Geometric Brownian motion for financial time series
- **Custom Mix**: Combination of multiple patterns

## 🔧 Configuration

All configuration is managed through `config.py` using Pydantic models:

```python
from config import config

# Model parameters
config.model.gru_units = 64
config.model.window_size = 20
config.model.epochs = 100

# Data parameters
config.data.time_steps = 1000
config.data.noise_level = 0.1
```

## Experiment Tracking

### MLflow Integration
- Automatic experiment logging
- Parameter and metric tracking
- Model versioning and artifacts

### Database Storage
- SQLite database for experiment metadata
- Training history per epoch
- Prediction results and errors
- Performance metrics comparison

## Interactive Features

### Streamlit Web UI
- Real-time data generation and visualization
- Interactive model training with progress tracking
- Comprehensive results analysis and comparison
- Experiment management dashboard

### Key UI Components
1. **Data Generation Tab**: Create and visualize time series data
2. **Model Training Tab**: Configure and train GRU models
3. **Predictions Tab**: Analyze model performance and predictions
4. **Experiments Tab**: Compare and manage multiple experiments
5. **Documentation Tab**: Comprehensive help and usage guide

## Performance Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination

## 🛠️ Advanced Features

### Callbacks and Monitoring
- Early stopping with patience
- Learning rate reduction on plateau
- Model checkpointing
- Custom database logging

### Data Preprocessing
- MinMax and Standard scaling options
- Sequence windowing for RNN input
- Train/validation/test splitting
- Inverse transformation for results

### Model Management
- Automatic model saving and loading
- Version control with timestamps
- Best model selection based on metrics
- Export capabilities for deployment

## Usage Examples

### Basic Training
```python
from gru_model import GRUTimeSeriesModel
from data_generator import TimeSeriesGenerator, DataPreprocessor

# Generate data
generator = TimeSeriesGenerator()
data = generator.generate_sine_wave(time_steps=1000, noise_level=0.1)

# Preprocess
preprocessor = DataPreprocessor()
scaled_data = preprocessor.fit_transform(data)
X, y = preprocessor.create_sequences(scaled_data, window_size=20)

# Train model
model = GRUTimeSeriesModel(input_shape=(20, 1))
results = model.train(X_train, y_train, experiment_name="my_experiment")
```

### Custom Architecture
```python
# Train with different architectures
architectures = ['standard', 'stacked', 'bidirectional', 'attention']

for arch in architectures:
    model = GRUTimeSeriesModel(input_shape=(20, 1))
    results = model.train(X_train, y_train, architecture=arch)
    metrics = model.evaluate(X_test, y_test)
    print(f"{arch}: R² = {metrics['r2']:.4f}")
```

## Monitoring and Logging

### Structured Logging
- Loguru-based logging with rotation
- Multiple log levels and outputs
- Training progress and error tracking

### Database Queries
```python
from database import db_manager

# Get experiment history
experiments = db_manager.get_experiments()
best_models = db_manager.get_best_experiments(metric='final_val_loss')
training_history = db_manager.get_training_history(experiment_id=1)
```

## Deployment Considerations

### Model Export
- TensorFlow SavedModel format
- MLflow model registry integration
- Containerization ready

### Scalability
- Configurable batch sizes
- GPU acceleration support
- Distributed training capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit for the interactive web framework
- MLflow for experiment tracking capabilities
- The open-source ML community for inspiration and tools

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the Streamlit app
- Review the code comments and docstrings


# GRU-Time-Series-Prediction
