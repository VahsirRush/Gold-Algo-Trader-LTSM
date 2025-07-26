"""
Enhanced Neural Network Strategy with Cross-Validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from .base import BaseStrategy
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

class EnhancedNNStrategy(BaseStrategy):
    def __init__(self, 
                 lookback_period: int = 30,
                 prediction_horizon: int = 5,
                 threshold: float = 0.015,
                 retrain_frequency: int = 63,
                 use_cross_validation: bool = True,
                 cv_folds: int = 5,
                 model_type: str = 'lstm'):
        """
        Initialize enhanced neural network strategy.
        """
        super().__init__('enhanced_nn')
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.threshold = threshold
        self.retrain_frequency = retrain_frequency
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.model_type = model_type
        
        # Model components
        self.model = None
        self.scaler = RobustScaler()
        self.last_retrain = 0
        self.cv_scores = []
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for EnhancedNNStrategy")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set."""
        df = data.copy()
        
        # Basic features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        
        # Volatility
        for period in [5, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # RSI
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['Close'], period)
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['Close'].rolling(bb_period).mean()
        bb_std_series = df['Close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_series * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_series * bb_std)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = self._calculate_atr(df, 14)
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = self._calculate_adx(df, 14)
        
        # Advanced features
        df['trend_strength'] = abs(df['sma_20'] - df['sma_100']) / df['sma_100']
        df['trend_direction'] = np.where(df['sma_20'] > df['sma_100'], 1, -1)
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        
        # +DI and -DI
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM/GRU models."""
        feature_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns']]
        
        # Handle categorical variables
        for col in feature_cols:
            if data[col].dtype == 'object':
                data[col] = pd.Categorical(data[col]).codes
        
        # Fill missing values
        data = data.fillna(method='ffill').fillna(0)
        
        # Prepare features and target
        X = data[feature_cols].values
        y = data['returns'].values
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(self.lookback_period, len(X) - self.prediction_horizon):
            X_seq.append(X[i-self.lookback_period:i])
            y_seq.append(y[i:i+self.prediction_horizon])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build neural network model."""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.prediction_horizon)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def cross_validate_model(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"  Cross-validation fold {fold + 1}/{self.cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build and train model
            model = self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_val, verbose=0)
            mse = mean_squared_error(y_val, y_pred)
            cv_scores.append(mse)
            
            print(f"    Fold {fold + 1} MSE: {mse:.6f}")
        
        return cv_scores
    
    def train_model(self, data: pd.DataFrame) -> bool:
        """Train the enhanced model with cross-validation."""
        try:
            print(f"Training enhanced {self.model_type.upper()} model...")
            
            # Create features
            feature_data = self.create_features(data)
            
            # Prepare sequences
            X, y = self.prepare_sequences(feature_data)
            
            if len(X) < 100:
                print("Insufficient data for training")
                return False
            
            # Cross-validation
            if self.use_cross_validation:
                print("Performing cross-validation...")
                self.cv_scores = self.cross_validate_model(X, y)
                cv_mean = np.mean(self.cv_scores)
                cv_std = np.std(self.cv_scores)
                print(f"Cross-validation MSE: {cv_mean:.6f} ± {cv_std:.6f}")
            
            # Train final model on all data
            print("Training final model...")
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=15,
                restore_best_weights=True
            )
            
            # Train
            history = self.model.fit(
                X, y,
                epochs=150,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            print(f"Model trained successfully!")
            print(f"Final loss: {history.history['loss'][-1]:.6f}")
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_returns(self, data: pd.DataFrame) -> np.ndarray:
        """Predict future returns."""
        if self.model is None:
            return np.zeros(len(data))
        
        try:
            # Create features
            feature_data = self.create_features(data)
            
            # Prepare sequences
            X, _ = self.prepare_sequences(feature_data)
            
            if len(X) == 0:
                return np.zeros(len(data))
            
            # Predict
            predictions = self.model.predict(X, verbose=0)
            
            # Return the first prediction (next day)
            return predictions[:, 0]
            
        except Exception as e:
            print(f"Error predicting: {e}")
            return np.zeros(len(data))
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on predicted returns."""
        try:
            # Train model if needed
            if self.model is None or len(data) - self.last_retrain > self.retrain_frequency:
                if self.train_model(data):
                    self.last_retrain = len(data)
            
            # Predict returns
            predicted_returns = self.predict_returns(data)
            
            # Generate signals based on predictions
            signals = pd.Series(0, index=data.index)
            
            # Convert predictions to signals
            for i, pred_return in enumerate(predicted_returns):
                if i + self.lookback_period < len(signals):
                    if pred_return > self.threshold:
                        signals.iloc[i + self.lookback_period] = 0.8
                    elif pred_return < -self.threshold:
                        signals.iloc[i + self.lookback_period] = -0.8
                    elif pred_return > self.threshold * 0.5:
                        signals.iloc[i + self.lookback_period] = 0.4
                    elif pred_return < -self.threshold * 0.5:
                        signals.iloc[i + self.lookback_period] = -0.4
            
            # Add confidence-based filtering
            confidence = abs(signals)
            signals[confidence < 0.3] = 0
            
            # Smooth signals
            signals = signals.rolling(window=3, min_periods=1).mean()
            
            self.signals = signals
            return signals
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            return pd.Series(0, index=data.index)
    
    def calculate_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on model predictions and cross-validation scores."""
        if self.model is None:
            return pd.Series(0.5, index=data.index)
        
        try:
            # Get predicted returns
            predicted_returns = self.predict_returns(data)
            
            # Calculate confidence based on prediction magnitude and CV scores
            prediction_confidence = np.clip(abs(predicted_returns) / 0.05, 0, 1)
            
            # Adjust confidence based on cross-validation performance
            if self.cv_scores:
                cv_confidence = 1 / (1 + np.mean(self.cv_scores))
                final_confidence = prediction_confidence * cv_confidence
            else:
                final_confidence = prediction_confidence
            
            # Pad with zeros for the lookback period
            confidence_series = pd.Series(0.0, index=data.index)
            confidence_series.iloc[self.lookback_period:self.lookback_period + len(final_confidence)] = final_confidence
            
            return confidence_series
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return pd.Series(0.5, index=data.index) 