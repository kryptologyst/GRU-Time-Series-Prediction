"""Main script for GRU Time Series Prediction with modern ML practices."""

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import sys
from pathlib import Path

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
logger.add("logs/gru_training.log", rotation="10 MB", level="DEBUG")

from config import config
from data_generator import TimeSeriesGenerator, DataPreprocessor, DataVisualizer
from gru_model import GRUTimeSeriesModel
from database import db_manager


def main():
    """Main execution function."""
    logger.info("Starting GRU Time Series Prediction")
    
    # Initialize components
    generator = TimeSeriesGenerator()
    preprocessor = DataPreprocessor()
    visualizer = DataVisualizer()
    
    try:
        # Generate synthetic data
        logger.info("Generating synthetic time series data")
        
        # Create multiple series for demonstration
        series_config = {
            'sine_wave': {
                'type': 'sine',
                'params': {'frequency': 1.5, 'amplitude': 1.0, 'noise_level': 0.1}
            },
            'trend_seasonal': {
                'type': 'trend_seasonal',
                'params': {'trend_slope': 0.01, 'seasonal_amplitude': 0.5, 'noise_level': 0.05}
            },
            'stock_like': {
                'type': 'stock',
                'params': {'initial_price': 100.0, 'volatility': 0.02, 'drift': 0.001}
            }
        }
        
        # Generate multiple series
        multi_series = generator.generate_multiple_series(series_config, config.data.time_steps)
        
        # Use combined series for training
        combined_data = multi_series['sine_wave'] + 0.3 * multi_series['trend_seasonal']
        
        # Visualize original data
        logger.info("Visualizing generated data")
        fig = visualizer.plot_time_series(
            {'Combined Series': combined_data},
            title="Generated Time Series Data"
        )
        plt.savefig(config.data_dir / "original_data.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Preprocess data
        logger.info("Preprocessing data")
        scaled_data = preprocessor.fit_transform(combined_data)
        
        # Create sequences
        X, y = preprocessor.create_sequences(scaled_data, config.model.window_size)
        logger.info(f"Created {len(X)} sequences with window size {config.model.window_size}")
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, config.model.train_split)
        
        # Reshape for GRU
        X_train = preprocessor.prepare_for_gru(X_train)
        X_test = preprocessor.prepare_for_gru(X_test)
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Test different architectures
        architectures = ['standard', 'stacked', 'bidirectional']
        results = {}
        
        for arch in architectures:
            logger.info(f"Training {arch} GRU model")
            
            # Create model
            model = GRUTimeSeriesModel(
                input_shape=(config.model.window_size, 1),
                model_config=config.model.dict()
            )
            
            # Train model
            experiment_name = f"GRU_{arch}_demo"
            training_results = model.train(
                X_train, y_train, X_test, y_test,
                experiment_name=experiment_name,
                architecture=arch
            )
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Make predictions for visualization
            y_pred = model.predict(X_test)
            y_test_orig = preprocessor.inverse_transform(y_test)
            y_pred_orig = preprocessor.inverse_transform(y_pred.flatten())
            
            # Store results
            results[arch] = {
                'model': model,
                'metrics': metrics,
                'predictions': (y_test_orig, y_pred_orig),
                'training_results': training_results
            }
            
            # Save model
            model_path = config.models_dir / f"gru_{arch}_model.h5"
            model.save_model(str(model_path))
            logger.info(f"Model saved to {model_path}")
            
            # Visualize predictions
            fig = visualizer.plot_predictions(
                y_test_orig, y_pred_orig,
                title=f"{arch.title()} GRU Predictions"
            )
            plt.savefig(config.data_dir / f"predictions_{arch}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"{arch} model - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.6f}")
        
        # Compare results
        logger.info("\n" + "="*50)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("="*50)
        
        for arch, result in results.items():
            metrics = result['metrics']
            training_time = result['training_results']['training_time']
            logger.info(f"{arch.upper()} GRU:")
            logger.info(f"  R²: {metrics['r2']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            logger.info(f"  MAE: {metrics['mae']:.6f}")
            logger.info(f"  Training Time: {training_time:.2f}s")
            logger.info("-" * 30)
        
        # Find best model
        best_arch = min(results.keys(), key=lambda x: results[x]['metrics']['rmse'])
        logger.info(f"Best performing model: {best_arch.upper()} (lowest RMSE)")
        
        # Generate comparison plot
        fig, axes = plt.subplots(len(architectures), 1, figsize=(12, 4*len(architectures)))
        if len(architectures) == 1:
            axes = [axes]
        
        for i, arch in enumerate(architectures):
            y_test_orig, y_pred_orig = results[arch]['predictions']
            
            axes[i].plot(y_test_orig[:100], label='Actual', linewidth=2, alpha=0.8)
            axes[i].plot(y_pred_orig[:100], label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
            axes[i].set_title(f'{arch.title()} GRU - R²: {results[arch]["metrics"]["r2"]:.4f}')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Step')
        plt.suptitle('GRU Architecture Comparison (First 100 Test Points)', fontsize=16)
        plt.tight_layout()
        plt.savefig(config.data_dir / "architecture_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Display experiment summary
        experiments_df = db_manager.get_experiments(limit=10)
        logger.info(f"\nTotal experiments in database: {len(experiments_df)}")
        
        if not experiments_df.empty:
            best_experiments = db_manager.get_best_experiments(limit=3)
            logger.info("\nTop 3 experiments by validation loss:")
            for _, exp in best_experiments.iterrows():
                logger.info(f"  ID {exp['id']}: {exp['name']} - Val Loss: {exp['final_val_loss']:.6f}")
        
        logger.info("\n🎉 GRU Time Series Prediction completed successfully!")
        logger.info(f"📊 Results saved to: {config.data_dir}")
        logger.info(f"🤖 Models saved to: {config.models_dir}")
        logger.info(f"📝 Logs saved to: {config.logs_dir}")
        logger.info("\n🚀 To run the interactive UI: streamlit run streamlit_app.py")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
