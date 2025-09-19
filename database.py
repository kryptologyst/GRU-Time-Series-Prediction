"""Database management for GRU experiments and results."""

import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import pandas as pd
from config import config

Base = declarative_base()


class Experiment(Base):
    """Model for storing experiment metadata."""
    
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Model parameters
    gru_units = Column(Integer)
    dropout_rate = Column(Float)
    recurrent_dropout = Column(Float)
    window_size = Column(Integer)
    
    # Training parameters
    epochs = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    
    # Results
    final_loss = Column(Float)
    final_val_loss = Column(Float)
    best_epoch = Column(Integer)
    training_time = Column(Float)  # in seconds
    
    # Status
    status = Column(String(50), default='running')  # running, completed, failed
    
    # Additional metadata as JSON
    experiment_metadata = Column(Text)  # JSON string for flexible storage


class TrainingHistory(Base):
    """Model for storing training history per epoch."""
    
    __tablename__ = 'training_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=False)
    epoch = Column(Integer, nullable=False)
    loss = Column(Float)
    val_loss = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """Model for storing predictions and actual values."""
    
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=False)
    time_step = Column(Integer, nullable=False)
    actual_value = Column(Float)
    predicted_value = Column(Float)
    error = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Database manager for GRU experiments."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or config.database_url
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def create_experiment(self, name: str, description: str = None, **kwargs) -> int:
        """Create a new experiment record."""
        session = self.get_session()
        try:
            experiment = Experiment(
                name=name,
                description=description,
                **kwargs
            )
            session.add(experiment)
            session.commit()
            experiment_id = experiment.id
            session.refresh(experiment)
            return experiment_id
        finally:
            session.close()
    
    def update_experiment(self, experiment_id: int, **kwargs):
        """Update an experiment record."""
        session = self.get_session()
        try:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                for key, value in kwargs.items():
                    if hasattr(experiment, key):
                        setattr(experiment, key, value)
                session.commit()
        finally:
            session.close()
    
    def log_training_history(self, experiment_id: int, epoch: int, loss: float, val_loss: float = None):
        """Log training history for an epoch."""
        session = self.get_session()
        try:
            history = TrainingHistory(
                experiment_id=experiment_id,
                epoch=epoch,
                loss=loss,
                val_loss=val_loss
            )
            session.add(history)
            session.commit()
        finally:
            session.close()
    
    def log_predictions(self, experiment_id: int, predictions_data: List[Dict[str, Any]]):
        """Log predictions for an experiment."""
        session = self.get_session()
        try:
            predictions = [
                Prediction(
                    experiment_id=experiment_id,
                    time_step=pred['time_step'],
                    actual_value=pred['actual'],
                    predicted_value=pred['predicted'],
                    error=pred['error']
                )
                for pred in predictions_data
            ]
            session.add_all(predictions)
            session.commit()
        finally:
            session.close()
    
    def get_experiments(self, limit: int = 100) -> pd.DataFrame:
        """Get all experiments as a DataFrame."""
        session = self.get_session()
        try:
            experiments = session.query(Experiment).order_by(Experiment.created_at.desc()).limit(limit).all()
            data = []
            for exp in experiments:
                data.append({
                    'id': exp.id,
                    'name': exp.name,
                    'description': exp.description,
                    'created_at': exp.created_at,
                    'gru_units': exp.gru_units,
                    'dropout_rate': exp.dropout_rate,
                    'window_size': exp.window_size,
                    'epochs': exp.epochs,
                    'batch_size': exp.batch_size,
                    'learning_rate': exp.learning_rate,
                    'final_loss': exp.final_loss,
                    'final_val_loss': exp.final_val_loss,
                    'best_epoch': exp.best_epoch,
                    'training_time': exp.training_time,
                    'status': exp.status
                })
            return pd.DataFrame(data)
        finally:
            session.close()
    
    def get_training_history(self, experiment_id: int) -> pd.DataFrame:
        """Get training history for an experiment."""
        session = self.get_session()
        try:
            history = session.query(TrainingHistory).filter(
                TrainingHistory.experiment_id == experiment_id
            ).order_by(TrainingHistory.epoch).all()
            
            data = []
            for h in history:
                data.append({
                    'epoch': h.epoch,
                    'loss': h.loss,
                    'val_loss': h.val_loss,
                    'timestamp': h.timestamp
                })
            return pd.DataFrame(data)
        finally:
            session.close()
    
    def get_predictions(self, experiment_id: int) -> pd.DataFrame:
        """Get predictions for an experiment."""
        session = self.get_session()
        try:
            predictions = session.query(Prediction).filter(
                Prediction.experiment_id == experiment_id
            ).order_by(Prediction.time_step).all()
            
            data = []
            for pred in predictions:
                data.append({
                    'time_step': pred.time_step,
                    'actual_value': pred.actual_value,
                    'predicted_value': pred.predicted_value,
                    'error': pred.error,
                    'created_at': pred.created_at
                })
            return pd.DataFrame(data)
        finally:
            session.close()
    
    def get_best_experiments(self, metric: str = 'final_val_loss', limit: int = 10) -> pd.DataFrame:
        """Get best experiments based on a metric."""
        session = self.get_session()
        try:
            if metric == 'final_val_loss':
                experiments = session.query(Experiment).filter(
                    Experiment.final_val_loss.isnot(None)
                ).order_by(Experiment.final_val_loss).limit(limit).all()
            elif metric == 'final_loss':
                experiments = session.query(Experiment).filter(
                    Experiment.final_loss.isnot(None)
                ).order_by(Experiment.final_loss).limit(limit).all()
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            data = []
            for exp in experiments:
                data.append({
                    'id': exp.id,
                    'name': exp.name,
                    'final_loss': exp.final_loss,
                    'final_val_loss': exp.final_val_loss,
                    'gru_units': exp.gru_units,
                    'dropout_rate': exp.dropout_rate,
                    'learning_rate': exp.learning_rate,
                    'training_time': exp.training_time
                })
            return pd.DataFrame(data)
        finally:
            session.close()


# Global database manager instance
db_manager = DatabaseManager()
