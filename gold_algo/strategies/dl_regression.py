"""
Enhanced Deep Learning Regression Strategy with Cross-Validation
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

class EnhancedDLRegressionStrategy(BaseStrategy):
    def __init__(self, 
                 lookback_period: int = 30,
                 prediction_horizon: int = 5,
                 threshold: float = 0.015,
                 retrain_frequency: int = 63,
                 use_cross_validation: bool = True,
                 cv_folds: int = 5,
                 use_advanced_features: bool = True,
                 model_type: str = 'lstm'):  # 'lstm', 'gru', 'transformer'
        """
        Initialize enhanced DL regression strategy.
        
        Args:
            lookback_period: Number of days to look back for features
            prediction_horizon: Number of days ahead to predict
            threshold: Minimum signal strength threshold
            retrain_frequency: How often to retrain the model
            use_cross_validation: Whether to use cross-validation
            cv_folds: Number of cross-validation folds
            use_advanced_features: Whether to use advanced feature engineering
            model_type: Type of neural network architecture
        """
        super().__init__('enhanced_dl_regression')
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.threshold = threshold
        self.retrain_frequency = retrain_frequency
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.use_advanced_features = use_advanced_features
        self.model_type = model_type
        
        # Model components
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = []
        self.last_retrain = 0
        self.cv_scores = []
        self.feature_importance = {}
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for EnhancedDLRegressionStrategy")
    
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set with advanced indicators."""
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Multiple timeframe moving averages
        for period in [3, 5, 8, 13, 21, 34, 55, 89]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
            df[f'price_ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}']
        
        # Volatility features
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_annualized_{period}'] = df[f'volatility_{period}'] * np.sqrt(252)
        
        # Volatility ratios and regimes
        df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
        df['vol_ratio_10_50'] = df['volatility_10'] / df['volatility_50']
        df['vol_regime'] = np.where(df['volatility_20'] > df['volatility_20'].rolling(100).mean() * 1.5, 'high',
                                  np.where(df['volatility_20'] < df['volatility_20'].rolling(100).mean() * 0.5, 'low', 'normal'))
        
        # Momentum features
        for period in [3, 5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'roc_{period}'] = df['Close'].pct_change(period)
            df[f'price_momentum_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
        
        # RSI variations
        for period in [7, 14, 21, 50]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['Close'], period)
        
        # MACD features
        for fast, slow in [(12, 26), (8, 21), (5, 13)]:
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
            df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
            df[f'macd_histogram_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}']
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_middle = df['Close'].rolling(period).mean()
            bb_std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
            df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_middle
        
        # ATR and ADX
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_ratio'] = df['atr_14'] / df['Close']
        df['adx_14'], df['plus_di_14'], df['minus_di_14'] = self._calculate_adx(df, 14)
        
        # Advanced features
        if self.use_advanced_features:
            # Regime features
            df['trend_strength'] = abs(df['sma_20'] - df['sma_100']) / df['sma_100']
            df['trend_direction'] = np.where(df['sma_20'] > df['sma_100'], 1, -1)
            df['trend_consistency'] = ((df['sma_20'] > df['sma_20'].shift(1)).rolling(10).sum() / 10)
            
            # Market efficiency
            df['efficiency_ratio'] = abs(df['Close'] - df['Close'].shift(20)) / df['atr_14'].rolling(20).sum()
            
            # Price patterns
            df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
            df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
            df['higher_close'] = (df['Close'] > df['Close'].shift(1)).astype(int)
            
            # Gap analysis
            df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
            df['gap_down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
            df['gap_size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            
            # Support and resistance
            df['support_level'] = df['Low'].rolling(20).min()
            df['resistance_level'] = df['High'].rolling(20).max()
            df['support_distance'] = (df['Close'] - df['support_level']) / df['Close']
            df['resistance_distance'] = (df['resistance_level'] - df['Close']) / df['Close']
            
            # Volume features (if available)
            if 'Volume' in df.columns:
                df['volume_sma_20'] = df['Volume'].rolling(20).mean()
                df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
                df['volume_price_trend'] = df['volume_ratio'] * df['returns']
        
        # Lagged features
        for lag in [1, 2, 3, 5, 8, 13]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            if 'Volume' in df.columns:
                df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        for period in [5, 10, 20]:
            df[f'returns_mean_{period}'] = df['returns'].rolling(period).mean()
            df[f'returns_std_{period}'] = df['returns'].rolling(period).std()
            df[f'returns_skew_{period}'] = df['returns'].rolling(period).skew()
            df[f'returns_kurt_{period}'] = df['returns'].rolling(period).kurt()
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
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
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = 'returns') -> Tuple[np.ndarray, np.ndarray]:
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
        y = data[target_col].values
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(self.lookback_period, len(X) - self.prediction_horizon):
            X_seq.append(X[i-self.lookback_period:i])
            y_seq.append(y[i:i+self.prediction_horizon])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build neural network model based on type."""
        if self.model_type == 'lstm':
            return self._build_lstm_model(input_shape)
        elif self.model_type == 'gru':
            return self._build_gru_model(input_shape)
        elif self.model_type == 'transformer':
            return self._build_transformer_model(input_shape)
        else:
            return self._build_lstm_model(input_shape)
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model."""
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
    
    def _build_gru_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build GRU model."""
        model = keras.Sequential([
            layers.GRU(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.GRU(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.GRU(32),
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
    
    def _build_transformer_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build Transformer model."""
        inputs = layers.Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=64
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = layers.LayerNormalization()(attention_output + inputs)
        
        # Feed forward
        ffn_output = layers.Dense(256, activation='relu')(attention_output)
        ffn_output = layers.Dense(input_shape[1])(ffn_output)
        
        # Add & Norm
        ffn_output = layers.LayerNormalization()(ffn_output + attention_output)
        
        # Global average pooling
        pooled = layers.GlobalAveragePooling1D()(ffn_output)
        
        # Dense layers
        dense = layers.Dense(64, activation='relu')(pooled)
        dense = layers.Dropout(0.2)(dense)
        dense = layers.Dense(32, activation='relu')(dense)
        outputs = layers.Dense(self.prediction_horizon)(dense)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
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
            feature_data = self.create_advanced_features(data)
            
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
            feature_data = self.create_advanced_features(data)
            
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