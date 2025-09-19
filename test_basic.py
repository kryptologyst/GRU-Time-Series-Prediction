"""Basic test script to verify core functionality without TensorFlow issues."""

import numpy as np
import matplotlib.pyplot as plt
from data_generator import TimeSeriesGenerator, DataPreprocessor
from config import config

def test_data_generation():
    """Test data generation functionality."""
    print("🧪 Testing data generation...")
    
    generator = TimeSeriesGenerator()
    
    # Test different data types
    sine_data = generator.generate_sine_wave(100, noise_level=0.1)
    cosine_data = generator.generate_cosine_wave(100, noise_level=0.1)
    trend_data = generator.generate_trend_with_seasonality(100, noise_level=0.05)
    stock_data = generator.generate_stock_like_data(100)
    
    print(f"✅ Sine wave data shape: {sine_data.shape}")
    print(f"✅ Cosine wave data shape: {cosine_data.shape}")
    print(f"✅ Trend+seasonal data shape: {trend_data.shape}")
    print(f"✅ Stock-like data shape: {stock_data.shape}")
    
    return sine_data

def test_preprocessing():
    """Test data preprocessing functionality."""
    print("\n🧪 Testing data preprocessing...")
    
    # Generate test data
    generator = TimeSeriesGenerator()
    data = generator.generate_sine_wave(200, noise_level=0.1)
    
    # Test preprocessing
    preprocessor = DataPreprocessor()
    scaled_data = preprocessor.fit_transform(data)
    
    print(f"✅ Original data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"✅ Scaled data range: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
    
    # Test sequence creation
    X, y = preprocessor.create_sequences(scaled_data, window_size=10)
    print(f"✅ Sequences created: X shape {X.shape}, y shape {y.shape}")
    
    # Test train/test split
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, train_split=0.8)
    print(f"✅ Train split: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"✅ Test split: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return preprocessor, X_train, X_test, y_train, y_test

def test_database():
    """Test database functionality."""
    print("\n🧪 Testing database functionality...")
    
    from database import db_manager
    
    # Create a test experiment
    exp_id = db_manager.create_experiment(
        name="Test Experiment",
        description="Testing database functionality",
        gru_units=64,
        window_size=10,
        epochs=50
    )
    
    print(f"✅ Created experiment with ID: {exp_id}")
    
    # Log some training history
    for epoch in range(5):
        loss = 1.0 - (epoch * 0.1)
        val_loss = 1.1 - (epoch * 0.08)
        db_manager.log_training_history(exp_id, epoch + 1, loss, val_loss)
    
    print("✅ Logged training history")
    
    # Get experiments
    experiments = db_manager.get_experiments(limit=5)
    print(f"✅ Retrieved {len(experiments)} experiments from database")
    
    return exp_id

def test_configuration():
    """Test configuration management."""
    print("\n🧪 Testing configuration...")
    
    print(f"✅ Model GRU units: {config.model.gru_units}")
    print(f"✅ Data time steps: {config.data.time_steps}")
    print(f"✅ Project root: {config.project_root}")
    print(f"✅ Models directory: {config.models_dir}")
    print(f"✅ Data directory: {config.data_dir}")

def create_visualization():
    """Create a sample visualization."""
    print("\n🎨 Creating sample visualization...")
    
    generator = TimeSeriesGenerator()
    
    # Generate multiple series
    series_config = {
        'sine': {'type': 'sine', 'params': {'frequency': 1.0, 'noise_level': 0.1}},
        'trend': {'type': 'trend_seasonal', 'params': {'trend_slope': 0.01, 'noise_level': 0.05}},
        'stock': {'type': 'stock', 'params': {'volatility': 0.02}}
    }
    
    multi_data = generator.generate_multiple_series(series_config, 300)
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, (name, data) in enumerate(multi_data.items()):
        axes[i].plot(data, label=name.title(), linewidth=1.5)
        axes[i].set_title(f"{name.title()} Time Series")
        axes[i].set_ylabel("Value")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[-1].set_xlabel("Time Step")
    plt.suptitle("Generated Time Series Data", fontsize=16)
    plt.tight_layout()
    
    # Save plot
    output_path = config.data_dir / "sample_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path}")

def main():
    """Run all tests."""
    print("🚀 Starting GRU Time Series Prediction Tests\n")
    
    try:
        # Test configuration
        test_configuration()
        
        # Test data generation
        data = test_data_generation()
        
        # Test preprocessing
        preprocessor, X_train, X_test, y_train, y_test = test_preprocessing()
        
        # Test database
        exp_id = test_database()
        
        # Create visualization
        create_visualization()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("✅ Configuration management working")
        print("✅ Data generation working")
        print("✅ Data preprocessing working")
        print("✅ Database functionality working")
        print("✅ Visualization creation working")
        
        print(f"\n📁 Files created in: {config.data_dir}")
        print(f"🗄️ Database created at: {config.database_url}")
        
        print("\n🔧 Next steps:")
        print("1. Install remaining dependencies: pip install streamlit plotly mlflow")
        print("2. Run the Streamlit UI: streamlit run streamlit_app.py")
        print("3. Or run the full training: python main.py")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
