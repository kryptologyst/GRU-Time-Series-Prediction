"""Modern GRU model implementation with advanced features."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l1_l2
try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
from typing import Tuple, Dict, Any, Optional, List
import time
import joblib
from pathlib import Path
from loguru import logger
from config import config
from database import db_manager


class GRUTimeSeriesModel:
    """Advanced GRU model for time series prediction with modern ML practices."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize GRU model.
        
        Args:
            input_shape: (time_steps, features)
            model_config: Model configuration parameters
        """
        self.input_shape = input_shape
        self.model_config = model_config or config.model.dict()
        self.model = None
        self.history = None
        self.experiment_id = None
        
        # Setup MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.experiment_name)
    
    def build_model(self, architecture: str = 'standard') -> Model:
        """
        Build GRU model with different architectures.
        
        Args:
            architecture: 'standard', 'stacked', 'bidirectional', 'attention'
        """
        if architecture == 'standard':
            return self._build_standard_gru()
        elif architecture == 'stacked':
            return self._build_stacked_gru()
        elif architecture == 'bidirectional':
            return self._build_bidirectional_gru()
        elif architecture == 'attention':
            return self._build_attention_gru()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _build_standard_gru(self) -> Model:
        """Build standard GRU model."""
        model = Sequential([
            GRU(
                self.model_config['gru_units'],
                activation='tanh',
                recurrent_activation='sigmoid',
                dropout=self.model_config['dropout_rate'],
                recurrent_dropout=self.model_config['recurrent_dropout'],
                input_shape=self.input_shape,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            ),
            BatchNormalization(),
            Dropout(self.model_config['dropout_rate']),
            Dense(32, activation='relu'),
            Dropout(self.model_config['dropout_rate'] / 2),
            Dense(1, activation='linear')
        ])
        
        return model
    
    def _build_stacked_gru(self) -> Model:
        """Build stacked GRU model."""
        model = Sequential([
            GRU(
                self.model_config['gru_units'],
                return_sequences=True,
                dropout=self.model_config['dropout_rate'],
                recurrent_dropout=self.model_config['recurrent_dropout'],
                input_shape=self.input_shape
            ),
            BatchNormalization(),
            GRU(
                self.model_config['gru_units'] // 2,
                dropout=self.model_config['dropout_rate'],
                recurrent_dropout=self.model_config['recurrent_dropout']
            ),
            BatchNormalization(),
            Dropout(self.model_config['dropout_rate']),
            Dense(32, activation='relu'),
            Dropout(self.model_config['dropout_rate'] / 2),
            Dense(1, activation='linear')
        ])
        
        return model
    
    def _build_bidirectional_gru(self) -> Model:
        """Build bidirectional GRU model."""
        from tensorflow.keras.layers import Bidirectional
        
        model = Sequential([
            Bidirectional(
                GRU(
                    self.model_config['gru_units'] // 2,
                    dropout=self.model_config['dropout_rate'],
                    recurrent_dropout=self.model_config['recurrent_dropout']
                ),
                input_shape=self.input_shape
            ),
            BatchNormalization(),
            Dropout(self.model_config['dropout_rate']),
            Dense(32, activation='relu'),
            Dropout(self.model_config['dropout_rate'] / 2),
            Dense(1, activation='linear')
        ])
        
        return model
    
    def _build_attention_gru(self) -> Model:
        """Build GRU model with attention mechanism."""
        from tensorflow.keras.layers import Attention, Concatenate, GlobalAveragePooling1D
        
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # GRU layer with return_sequences=True for attention
        gru_out = GRU(
            self.model_config['gru_units'],
            return_sequences=True,
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=self.model_config['recurrent_dropout']
        )(inputs)
        
        # Attention mechanism (simplified)
        attention_weights = Dense(1, activation='tanh')(gru_out)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        
        # Apply attention weights
        context_vector = tf.reduce_sum(gru_out * attention_weights, axis=1)
        
        # Final layers
        x = BatchNormalization()(context_vector)
        x = Dropout(self.model_config['dropout_rate'])(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.model_config['dropout_rate'] / 2)(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_model(self, 
                     optimizer: str = 'adam',
                     loss: str = 'mse',
                     metrics: List[str] = None):
        """Compile the model with specified optimizer and loss."""
        if metrics is None:
            metrics = ['mae', 'mse']
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=self.model_config['learning_rate'])
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=self.model_config['learning_rate'])
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with optimizer: {optimizer}, loss: {loss}")
    
    def get_callbacks(self, model_name: str = "gru_model") -> List:
        """Get training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.model_config['patience'],
            min_delta=self.model_config['min_delta'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.model_config['patience'] // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = config.models_dir / f"{model_name}_best.h5"
        model_checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        return callbacks
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              experiment_name: str = None,
              architecture: str = 'standard') -> Dict[str, Any]:
        """
        Train the GRU model with MLflow tracking.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            experiment_name: Name for the experiment
            architecture: Model architecture to use
        
        Returns:
            Training history and metrics
        """
        start_time = time.time()
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            mlflow_run = mlflow.start_run(run_name=experiment_name)
            # Log parameters
            mlflow.log_params(self.model_config)
            mlflow.log_param("architecture", architecture)
            mlflow.log_param("input_shape", self.input_shape)
        else:
            mlflow_run = None
        
        try:
            # Create experiment in database
            self.experiment_id = db_manager.create_experiment(
                name=experiment_name or f"GRU_{architecture}_{int(time.time())}",
                description=f"GRU model with {architecture} architecture",
                **self.model_config
            )
            
            # Build and compile model
            self.model = self.build_model(architecture)
            self.compile_model()
            
            # Log model summary if MLflow is available
            if MLFLOW_AVAILABLE:
                model_summary = []
                self.model.summary(print_fn=lambda x: model_summary.append(x))
                mlflow.log_text('\n'.join(model_summary), "model_summary.txt")
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            else:
                # Use validation split
                validation_data = None
            
            # Get callbacks
            callbacks = self.get_callbacks(f"gru_{architecture}_{self.experiment_id}")
            
            # Custom callback for logging to database
            class DatabaseCallback(tf.keras.callbacks.Callback):
                def __init__(self, experiment_id):
                    self.experiment_id = experiment_id
                
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    db_manager.log_training_history(
                        self.experiment_id,
                        epoch + 1,
                        logs.get('loss'),
                        logs.get('val_loss')
                    )
            
            callbacks.append(DatabaseCallback(self.experiment_id))
            
            # Train model
            logger.info(f"Starting training for experiment {self.experiment_id}")
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.model_config['epochs'],
                batch_size=self.model_config['batch_size'],
                validation_data=validation_data,
                validation_split=self.model_config['validation_split'] if validation_data is None else None,
                callbacks=callbacks,
                verbose=1
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Log metrics
            final_loss = min(self.history.history['loss'])
            final_val_loss = min(self.history.history.get('val_loss', [float('inf')]))
            best_epoch = np.argmin(self.history.history.get('val_loss', self.history.history['loss'])) + 1
            
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("final_loss", final_loss)
                mlflow.log_metric("final_val_loss", final_val_loss)
                mlflow.log_metric("best_epoch", best_epoch)
                mlflow.log_metric("training_time", training_time)
            
            # Update experiment in database
            db_manager.update_experiment(
                self.experiment_id,
                final_loss=final_loss,
                final_val_loss=final_val_loss,
                best_epoch=best_epoch,
                training_time=training_time,
                status='completed'
            )
            
            # Log model if MLflow is available
            if MLFLOW_AVAILABLE:
                mlflow.tensorflow.log_model(self.model, "model")
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Final loss: {final_loss:.6f}, Final val_loss: {final_val_loss:.6f}")
            
            return {
                'history': self.history.history,
                'final_loss': final_loss,
                'final_val_loss': final_val_loss,
                'best_epoch': best_epoch,
                'training_time': training_time,
                'experiment_id': self.experiment_id
            }
        
        finally:
            # End MLflow run if it was started
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.end_run()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_test - y_pred.flatten()) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        # Log predictions to database if experiment exists
        if self.experiment_id:
            predictions_data = [
                {
                    'time_step': i,
                    'actual': float(y_test[i]),
                    'predicted': float(y_pred[i, 0]),
                    'error': float(y_test[i] - y_pred[i, 0])
                }
                for i in range(len(y_test))
            ]
            db_manager.log_predictions(self.experiment_id, predictions_data)
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "No model built yet"
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
