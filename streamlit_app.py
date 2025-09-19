"""Modern Streamlit UI for GRU Time Series Prediction."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json

# Import our modules
from config import config
from data_generator import TimeSeriesGenerator, DataPreprocessor, DataVisualizer
from gru_model import GRUTimeSeriesModel
from database import db_manager

# Page configuration
st.set_page_config(
    page_title="GRU Time Series Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

# Main header
st.markdown('<h1 class="main-header">🧠 GRU Time Series Prediction</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data Generation Settings
    st.subheader("📊 Data Settings")
    data_type = st.selectbox(
        "Data Type",
        ["Sine Wave", "Cosine Wave", "Trend + Seasonality", "Stock-like", "Custom Mix"]
    )
    
    time_steps = st.slider("Time Steps", 100, 2000, config.data.time_steps)
    noise_level = st.slider("Noise Level", 0.0, 0.5, config.data.noise_level, 0.01)
    
    # Model Settings
    st.subheader("🤖 Model Settings")
    architecture = st.selectbox(
        "Architecture",
        ["standard", "stacked", "bidirectional", "attention"]
    )
    
    gru_units = st.slider("GRU Units", 16, 128, config.model.gru_units)
    window_size = st.slider("Window Size", 5, 50, config.model.window_size)
    epochs = st.slider("Epochs", 10, 200, config.model.epochs)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=config.model.learning_rate
    )

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Generation", "🏋️ Model Training", "📈 Predictions", "📋 Experiments", "📚 Documentation"])

with tab1:
    st.header("📊 Data Generation & Visualization")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Generate Data")
        
        if st.button("🎲 Generate New Data", type="primary"):
            with st.spinner("Generating time series data..."):
                generator = TimeSeriesGenerator()
                
                if data_type == "Sine Wave":
                    data = generator.generate_sine_wave(
                        time_steps=time_steps,
                        noise_level=noise_level
                    )
                elif data_type == "Cosine Wave":
                    data = generator.generate_cosine_wave(
                        time_steps=time_steps,
                        noise_level=noise_level
                    )
                elif data_type == "Trend + Seasonality":
                    data = generator.generate_trend_with_seasonality(
                        time_steps=time_steps,
                        noise_level=noise_level
                    )
                elif data_type == "Stock-like":
                    data = generator.generate_stock_like_data(
                        time_steps=time_steps
                    )
                else:  # Custom Mix
                    series_config = {
                        'sine': {'type': 'sine', 'params': {'frequency': 1.0, 'noise_level': noise_level}},
                        'trend': {'type': 'trend_seasonal', 'params': {'noise_level': noise_level}},
                    }
                    multi_data = generator.generate_multiple_series(series_config, time_steps)
                    data = multi_data['sine'] + 0.3 * multi_data['trend']
                
                st.session_state.current_data = data
                st.session_state.data_generated = True
                st.success("✅ Data generated successfully!")
        
        # Data statistics
        if st.session_state.data_generated and st.session_state.current_data is not None:
            st.subheader("📊 Data Statistics")
            data = st.session_state.current_data
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Mean", f"{np.mean(data):.4f}")
                st.metric("Min", f"{np.min(data):.4f}")
            with col_stat2:
                st.metric("Std Dev", f"{np.std(data):.4f}")
                st.metric("Max", f"{np.max(data):.4f}")
    
    with col1:
        if st.session_state.data_generated and st.session_state.current_data is not None:
            st.subheader("📈 Time Series Visualization")
            
            # Create interactive plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.current_data,
                mode='lines',
                name='Time Series',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"{data_type} Time Series Data",
                xaxis_title="Time Step",
                yaxis_title="Value",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plot
            st.subheader("📊 Value Distribution")
            fig_hist = px.histogram(
                x=st.session_state.current_data,
                nbins=50,
                title="Value Distribution"
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.header("🏋️ Model Training")
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first in the Data Generation tab.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Training Configuration")
            
            experiment_name = st.text_input(
                "Experiment Name",
                value=f"GRU_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            train_split = st.slider("Training Split", 0.6, 0.9, config.model.train_split)
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training GRU model..."):
                    # Prepare data
                    preprocessor = DataPreprocessor()
                    scaled_data = preprocessor.fit_transform(st.session_state.current_data)
                    
                    # Create sequences
                    X, y = preprocessor.create_sequences(scaled_data, window_size)
                    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, train_split)
                    
                    # Reshape for GRU
                    X_train = preprocessor.prepare_for_gru(X_train)
                    X_test = preprocessor.prepare_for_gru(X_test)
                    
                    # Update model config
                    model_config = {
                        'gru_units': gru_units,
                        'window_size': window_size,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'dropout_rate': config.model.dropout_rate,
                        'recurrent_dropout': config.model.recurrent_dropout,
                        'patience': config.model.patience,
                        'min_delta': config.model.min_delta,
                        'train_split': train_split,
                        'validation_split': config.model.validation_split
                    }
                    
                    # Create and train model
                    model = GRUTimeSeriesModel(
                        input_shape=(window_size, 1),
                        model_config=model_config
                    )
                    
                    # Train with progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    training_results = model.train(
                        X_train, y_train, X_test, y_test,
                        experiment_name=experiment_name,
                        architecture=architecture
                    )
                    
                    # Store results
                    st.session_state.trained_model = model
                    st.session_state.training_results = training_results
                    st.session_state.model_trained = True
                    st.session_state.preprocessor = preprocessor
                    st.session_state.test_data = (X_test, y_test)
                    
                    progress_bar.progress(100)
                    status_text.success("✅ Training completed!")
        
        with col1:
            if st.session_state.model_trained and st.session_state.training_results:
                st.subheader("📊 Training Results")
                
                results = st.session_state.training_results
                
                # Metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Final Loss", f"{results['final_loss']:.6f}")
                with col_m2:
                    st.metric("Val Loss", f"{results['final_val_loss']:.6f}")
                with col_m3:
                    st.metric("Best Epoch", results['best_epoch'])
                with col_m4:
                    st.metric("Training Time", f"{results['training_time']:.1f}s")
                
                # Training history plot
                st.subheader("📈 Training History")
                history = results['history']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['loss'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='#1f77b4')
                ))
                
                if 'val_loss' in history:
                    fig.add_trace(go.Scatter(
                        y=history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='#ff7f0e')
                    ))
                
                fig.update_layout(
                    title="Training History",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("📈 Predictions & Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the Model Training tab.")
    else:
        # Make predictions
        model = st.session_state.trained_model
        preprocessor = st.session_state.preprocessor
        X_test, y_test = st.session_state.test_data
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform
        y_test_orig = preprocessor.inverse_transform(y_test)
        y_pred_orig = preprocessor.inverse_transform(y_pred.flatten())
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        # Display metrics
        st.subheader("📊 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MSE", f"{metrics['mse']:.6f}")
        with col2:
            st.metric("MAE", f"{metrics['mae']:.6f}")
        with col3:
            st.metric("RMSE", f"{metrics['rmse']:.6f}")
        with col4:
            st.metric("R²", f"{metrics['r2']:.4f}")
        
        # Predictions plot
        st.subheader("🎯 Predictions vs Actual")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Time Series Comparison', 'Actual vs Predicted Scatter'),
            vertical_spacing=0.1
        )
        
        # Time series plot
        fig.add_trace(
            go.Scatter(
                y=y_test_orig,
                mode='lines',
                name='Actual',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=y_pred_orig,
                mode='lines',
                name='Predicted',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_test_orig,
                y=y_pred_orig,
                mode='markers',
                name='Predictions',
                marker=dict(color='#2ca02c', size=6, opacity=0.6)
            ),
            row=2, col=1
        )
        
        # Perfect prediction line
        min_val, max_val = min(y_test_orig.min(), y_pred_orig.min()), max(y_test_orig.max(), y_pred_orig.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Time Step", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="Actual Values", row=2, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        st.subheader("🔍 Error Analysis")
        errors = y_test_orig - y_pred_orig
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            fig_error = px.histogram(
                x=errors,
                nbins=30,
                title="Prediction Error Distribution"
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        with col2:
            # Error over time
            fig_error_time = go.Figure()
            fig_error_time.add_trace(go.Scatter(
                y=errors,
                mode='lines+markers',
                name='Prediction Error',
                line=dict(color='red')
            ))
            fig_error_time.update_layout(
                title="Prediction Error Over Time",
                xaxis_title="Time Step",
                yaxis_title="Error"
            )
            st.plotly_chart(fig_error_time, use_container_width=True)

with tab4:
    st.header("📋 Experiment Management")
    
    # Get experiments from database
    experiments_df = db_manager.get_experiments()
    
    if not experiments_df.empty:
        st.subheader("🔬 Recent Experiments")
        
        # Display experiments table
        st.dataframe(
            experiments_df[['id', 'name', 'created_at', 'gru_units', 'final_val_loss', 'training_time', 'status']],
            use_container_width=True
        )
        
        # Best experiments
        st.subheader("🏆 Best Performing Models")
        best_experiments = db_manager.get_best_experiments(limit=5)
        
        if not best_experiments.empty:
            st.dataframe(best_experiments, use_container_width=True)
        
        # Experiment details
        st.subheader("📊 Experiment Analysis")
        
        if len(experiments_df) > 0:
            selected_exp_id = st.selectbox(
                "Select Experiment",
                experiments_df['id'].tolist(),
                format_func=lambda x: f"ID {x}: {experiments_df[experiments_df['id']==x]['name'].iloc[0]}"
            )
            
            if selected_exp_id:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Training history
                    history_df = db_manager.get_training_history(selected_exp_id)
                    if not history_df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=history_df['epoch'],
                            y=history_df['loss'],
                            mode='lines',
                            name='Training Loss'
                        ))
                        if 'val_loss' in history_df.columns and history_df['val_loss'].notna().any():
                            fig.add_trace(go.Scatter(
                                x=history_df['epoch'],
                                y=history_df['val_loss'],
                                mode='lines',
                                name='Validation Loss'
                            ))
                        fig.update_layout(title=f"Training History - Experiment {selected_exp_id}")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Predictions
                    predictions_df = db_manager.get_predictions(selected_exp_id)
                    if not predictions_df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=predictions_df['time_step'],
                            y=predictions_df['actual_value'],
                            mode='lines',
                            name='Actual'
                        ))
                        fig.add_trace(go.Scatter(
                            x=predictions_df['time_step'],
                            y=predictions_df['predicted_value'],
                            mode='lines',
                            name='Predicted'
                        ))
                        fig.update_layout(title=f"Predictions - Experiment {selected_exp_id}")
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No experiments found. Train a model to see experiment results here.")

with tab5:
    st.header("📚 Documentation & Help")
    
    st.markdown("""
    ## 🧠 GRU Time Series Prediction
    
    This application demonstrates advanced GRU (Gated Recurrent Unit) networks for time series prediction with modern ML practices.
    
    ### 🚀 Features
    
    - **Multiple Architectures**: Standard, Stacked, Bidirectional, and Attention-based GRU models
    - **Data Generation**: Various synthetic time series patterns (sine, cosine, trend+seasonality, stock-like)
    - **Modern ML Stack**: TensorFlow 2.x, MLflow tracking, experiment management
    - **Interactive UI**: Real-time visualization and model interaction
    - **Experiment Tracking**: SQLite database for storing experiments and results
    
    ### 📊 How to Use
    
    1. **Generate Data**: Choose a data type and generate synthetic time series data
    2. **Configure Model**: Select architecture and hyperparameters
    3. **Train Model**: Start training with experiment tracking
    4. **Analyze Results**: View predictions, metrics, and error analysis
    5. **Compare Experiments**: Track and compare multiple experiments
    
    ### 🏗️ Architecture Options
    
    - **Standard**: Single GRU layer with dense output
    - **Stacked**: Multiple GRU layers for complex patterns
    - **Bidirectional**: Processes sequences in both directions
    - **Attention**: Attention mechanism for better long-term dependencies
    
    ### 📈 Metrics Explained
    
    - **MSE**: Mean Squared Error - average squared differences
    - **MAE**: Mean Absolute Error - average absolute differences  
    - **RMSE**: Root Mean Squared Error - square root of MSE
    - **R²**: Coefficient of determination - proportion of variance explained
    
    ### 🔧 Technical Stack
    
    - **TensorFlow 2.x**: Deep learning framework
    - **Streamlit**: Interactive web application
    - **Plotly**: Interactive visualizations
    - **MLflow**: Experiment tracking and model management
    - **SQLAlchemy**: Database ORM for experiment storage
    - **Pydantic**: Configuration management
    
    ### 📝 Best Practices Implemented
    
    - Configuration management with Pydantic
    - Experiment tracking with MLflow and database
    - Early stopping and learning rate scheduling
    - Model checkpointing and versioning
    - Comprehensive logging and error handling
    - Interactive visualization and analysis
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🧠 Advanced GRU Time Series Prediction | Built with Streamlit & TensorFlow"
    "</div>",
    unsafe_allow_html=True
)
