import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf
 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Local imports
# from nbeats_block import NBeatsBlock

import requests
from streamlit_lottie import st_lottie_spinner

import plotly.graph_objects as go

@st.cache_data
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://lottie.host/5ff47963-f991-4885-b937-33d91d42e1ff/pZDr4Reauz.json"
lottie_json = load_lottieurl(lottie_url)

# Global hyperparameter
TICKER = ''
N_SPLIT = 0.9

# Hyperparameter utama penelitian
WINDOWS = 0
HORIZON = 0
BATCH_SIZE = 64

# Hyperparamter model N-BEATS
N_LAYERS = 4
N_STACKS = 30
N_NEURONS = 512
N_EPOCHS = 5000
INPUT_SIZE = None
THETA_SIZE = None

# Lainnya
N_PATIENCE_ES = 150
N_PATIENCE_LR = 150 

# Save path 
SPLIT_STR = int(N_SPLIT * 100) 
BASE_PATH = "saved_model"
BASELINE_PATH = ''
LSTM_PATH = ''
NBEATS_PATH = ''

def predict_contents(df, ticker, periods):
    # 1. Menyisakan kolom date dan price
    df_date_price = pd.DataFrame(df["Close"]).rename(columns={"Close": "Price"})

    # 2. Forward filling: mengisi row (price) yang kosong dengan row sebelumnya
    if df_date_price["Price"].isnull().sum(axis=0) > 0:
        df_date_price.ffill(axis=0)

    visualize_prices(df_date_price, ticker)

    # create_lstm_prediction(df_date_price, ticker, periods)
    with st.expander("LSTM Model Prediction"):
        with st_lottie_spinner(lottie_json, height=100):
            lstm_error = create_lstm_prediction(df_date_price, ticker, periods)

    with st.expander("N-BEATS Model Prediction"):
        with st_lottie_spinner(lottie_json, height=100):
            nbeats_error = create_nbeats_model(df_date_price, ticker, periods)

    with st.expander("Summary"):
        with st_lottie_spinner(lottie_json, height=100):
            get_summary(lstm_error, nbeats_error, periods)
      
def visualize_prices(df_prices, ticker):
    with st.expander(f"{ticker} Price Visualization"):
        st.line_chart(df_prices)

def visualize_naive(timesteps, data, data_actual):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timesteps, y=data_actual, name="Actual", marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=timesteps[1:], y=data, name="Prediction", marker=dict(color='red')))

    st.write(fig)

def visualize_pred_actual(timesteps, data, data_actual, start=0, end=None):
    if len(timesteps) == 1 or len(data) == 1 or len(data_actual) == 1:
        data_timesteps = timesteps
        pred_prices = [data]
        actual_prices = data_actual
    else:
        data_timesteps = timesteps[start:end]
        pred_prices = data[start:end]
        actual_prices = data_actual[start:end]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_timesteps, y=pred_prices, name="Prediction", marker=dict(color='blue'), mode="markers"))
    fig.add_trace(go.Scatter(x=data_timesteps, y=actual_prices, name="Actual", marker=dict(color='orange')))

    st.write(fig)

def create_windows_horizons(data, window=WINDOWS, horizon=HORIZON):
    window_step = np.expand_dims(np.arange(window + horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(data) - (window + horizon - 1)), axis=0).T
    windowed_prices = data[window_indexes]
    windows, horizons = windowed_prices[:, :-horizon], windowed_prices[:, -horizon:]

    return windows, horizons

# Membuat metrik tingkat kesalahan prediksi (error)
def create_error_metrics(y_actual, y_pred):
    # Konversi data ke tipe float32 (tipe data default Tensorflow)
    y_actual = tf.cast(y_actual, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mse = tf.keras.metrics.mean_squared_error(y_actual, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_actual, y_pred)

    # Jika data hasil prediksi lebih dari 1, maka diambil nilai rata-rata nya
    if tf.size(y_pred) > 0:
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)

    return {
        "RMSE": rmse.numpy(),
        "MAPE": mape.numpy()
    }

def save_model_checkpoint(model_name, save_path="model_checkpoints"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), monitor="val_loss", verbose=0, save_best_only=True)

def create_dataset_splits(windows, horizons, train_size=N_SPLIT):
    split_len = int(len(windows) * train_size)
    windows_train = windows[:split_len]
    horizons_train = horizons[:split_len]
    windows_test = windows[split_len:]
    horizons_test = horizons[split_len:]

    return windows_train, windows_test, horizons_train, horizons_test

def create_prediction(model, test_data):
    pred_result = model.predict(test_data)
    return tf.squeeze(pred_result)

def model_options(ticker):
    if ticker == 'BTC':
        return {
            'windows': 7,
            'horizon': 1,
            'batch_size': 128
        }
    elif ticker == 'USDC':
        return {
            'windows': 7,
            'horizon': 1,
            'batch_size': 32
        }
    elif ticker == 'XRP':
        return {
            'windows': 7,
            'horizon': 1,
            'batch_size': 64
        }

def create_lstm_prediction(df_date_price, ticker, periods):
    options = model_options(ticker)
    windows = options['windows']
    horizon = options['horizon']
    batch_size = options['batch_size']

    LSTM_PATH = f"{BASE_PATH}_lstm/lstm_model_{ticker}_{windows}_{horizon}_{batch_size}_{SPLIT_STR}.keras"

    # Memisahkan data date dan price ke array masing-masing
    timesteps = df_date_price.index.to_numpy()[windows:]
    prices = df_date_price["Price"].to_numpy()[windows:]

    split_len = int(len(prices) * N_SPLIT)

    # X train dan test berisi data timesteps
    # y train dan test berisi data harga crypto
    X_train, X_test = timesteps[:split_len], timesteps[split_len:]
    y_train, y_test = prices[:split_len], prices[split_len:]

    all_windows, all_horizons = create_windows_horizons(prices, window=windows, horizon=horizon)
    windows_train, windows_test, horizons_train, horizons_test = create_dataset_splits(all_windows, all_horizons)

    if os.path.exists(LSTM_PATH):
        lstm_model = load_model(LSTM_PATH)
    else:
        tf.random.set_seed(42)

        # 1. Membangun Model LSTM 
        input_layers = layers.Input(shape=windows)
        x = layers.Lambda(lambda x : tf.expand_dims(x, axis=1))(input_layers)
        x = layers.LSTM(128, activation="relu")(x)
        output = layers.Dense(horizon)(x)
        lstm_model = Model(input_layers, outputs=output, name="lstm_model")

        # 2. Compile model LSTM
        lstm_model.compile(loss="mae",
                        optimizer=Adam()) 
        
        lstm_model.fit(windows_train,
                    horizons_train,
                    epochs=N_EPOCHS,
                    verbose=0,
                    batch_size=BATCH_SIZE,
                    validation_data=(windows_test, horizons_test),
                    callbacks=[save_model_checkpoint(model_name=lstm_model.name),
                                EarlyStopping(monitor="val_loss",
                                                patience=N_PATIENCE_ES,
                                                restore_best_weights=True),
                                ReduceLROnPlateau(monitor="val_loss",
                                                    patience=N_PATIENCE_LR,
                                                    verbose=1)])
        
        lstm_model = load_model("model_checkpoints/lstm_model")
        lstm_model.save(LSTM_PATH) 
    
    lstm_pred = create_prediction(lstm_model, windows_test[:periods])
    
    if lstm_pred.ndim > 1:
        lstm_preds = tf.reduce_mean(lstm_pred, axis=1)
    else:
        lstm_preds = lstm_pred
    
    lstm_res = create_error_metrics(y_actual=tf.squeeze(horizons_test)[:periods], y_pred=lstm_preds)

    visualize_pred_actual(timesteps=X_test[:periods],
                          data=lstm_preds,
                          data_actual=horizons_test[:, 0][:periods])

    return lstm_res

block_name = "CustomLayers>NBeatsBlock"
    
custom_objects = tf.keras.utils.get_custom_objects()
if block_name in custom_objects:
    pass
else:
    @tf.keras.utils.register_keras_serializable(package="CustomLayers")
    class NBeatsBlock(tf.keras.layers.Layer):
        def __init__(self,
                    input_size: int,
                    theta_size: int,
                    horizon: int,
                    n_neurons: int,
                    n_layers: int,
                    **kwargs):
            super().__init__(**kwargs)
            self.input_size = input_size
            self.theta_size = theta_size
            self.horizon = horizon
            self.n_neurons = n_neurons
            self.n_layers = n_layers

            # Hidden layer -> FC Stack yang terdiri dari 4 layers
            self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for i in range(n_layers)]
            # Theta layer -> hasil output dengan linear activation
            self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

        # Fungsi call dijalankan ketika layer dipanggil
        def call(self, inputs):
            x = inputs
            for layer in self.hidden:
                x = layer(x)
            theta = self.theta_layer(x)
            # Hasil output dari theta layer berupa (backcast, forecast)
            backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
            return backcast, forecast
        
        def get_config(self):
            config = {
                'input_size': self.input_size,
                'theta_size': self.theta_size,
                'horizon': self.horizon,
                'n_neurons': self.n_neurons,
                'n_layers': self.n_layers,
            }
            base_config = super(NBeatsBlock, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
            
def create_nbeats_model(df_date_price, ticker, periods):   
    options = model_options(ticker)
    windows = options['windows']
    horizon = options['horizon']
    batch_size = options['batch_size']

    NBEATS_PATH = f"{BASE_PATH}_nbeats/nbeats_model_{ticker}_{windows}_{horizon}_{batch_size}_{SPLIT_STR}.keras"

    date_price_nbeats = df_date_price.copy()
    for i in range(windows):
        date_price_nbeats[f"Price+{i+1}"] = date_price_nbeats["Price"].shift(periods=i+1)
        
    X = date_price_nbeats.dropna().drop("Price", axis=1)
    y = date_price_nbeats.dropna()["Price"]

    split_len = int(len(X) * N_SPLIT)
    X_train, y_train = X[:split_len], y[:split_len]
    X_test, y_test = X[split_len:], y[split_len:]

    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

    test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

    train_dataset_zipped = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
    test_dataset_zipped = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

    BATCH_SIZE_ZIPPED = 1024

    train_dataset = train_dataset_zipped.batch(BATCH_SIZE_ZIPPED).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset_zipped.batch(BATCH_SIZE_ZIPPED).prefetch(tf.data.AUTOTUNE)

    INPUT_SIZE = windows
    THETA_SIZE = INPUT_SIZE + horizon
    
    if os.path.exists(NBEATS_PATH):
        nbeats_model = load_model(NBEATS_PATH)
    else:
        # Pembuatan Model
        tf.random.set_seed(42)

        block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                    theta_size=THETA_SIZE,
                                    horizon=horizon,
                                    n_neurons=N_NEURONS,
                                    n_layers=N_LAYERS,
                                    name="basic_block")
        
        stack_input = layers.Input(shape=(INPUT_SIZE), name="stack_input")

        residuals, forecast = block_layer(stack_input)

        for (i, j) in enumerate(range(N_STACKS - 1)):
            backcast, block_forecast = NBeatsBlock(input_size=INPUT_SIZE,
                                                    theta_size=THETA_SIZE,
                                                    horizon=horizon,
                                                    n_neurons=N_NEURONS,
                                                    n_layers=N_LAYERS,
                                                    name=f"NBeatsBlock-{i}")(residuals)
            
            residuals = layers.subtract([residuals, backcast], name=f"Subtract-{i}")
            forecast = layers.add([forecast, block_forecast], name=f"Add-{i}")

        nbeats_model = Model(inputs=stack_input, outputs=forecast, name="nbeats_model")

        nbeats_model.compile(loss="mae",
                                optimizer=Adam())
        
        nbeats_model.fit(train_dataset,
                         epochs=N_EPOCHS,
                         validation_data=test_dataset,
                         verbose=0, 
                         batch_size=BATCH_SIZE,
                         callbacks=[save_model_checkpoint(model_name=nbeats_model.name),
                                    tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=150,
                                                            restore_best_weights=True),
                                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                        patience=150,
                                                        verbose=1)])       
        
        nbeats_model = load_model("model_checkpoints/nbeats_model")
        nbeats_model.save(NBEATS_PATH)
                
    nbeats_pred = create_prediction(nbeats_model, X_test[:periods])

    if nbeats_pred.ndim > 1:
        nbeats_preds = tf.reduce_mean(nbeats_pred, axis=1)
    else:
        nbeats_preds = nbeats_pred

    nbeats_res = create_error_metrics(y_actual=y_test[:periods], y_pred=nbeats_preds)

    visualize_pred_actual(timesteps=y_test.index.values[:periods],
                          data=nbeats_preds,
                          data_actual=y_test[:periods])
    
    return nbeats_res
    
    
def get_summary(errors_lstm, errors_nbeats, periods):
    mape_lstm = errors_lstm["MAPE"]
    mape_nbeats = errors_nbeats["MAPE"]

    if mape_lstm < mape_nbeats:
        st.write("Based on the prediction error percentage, the **LSTM** prediction is *more accurate* than the **N-BEATS**.")
    elif mape_lstm > mape_nbeats:
        st.write("Based on the prediction error percentage, the **N-BEATS** prediction is *more accurate* than the **LSTM**.")
    st.write(f"LSTM model yields an error percentage of {'{:10.4f}'.format(mape_lstm)}%")
    st.write(f"N-BEATS model yields an error percentage of {'{:10.4f}'.format(mape_nbeats)}%")