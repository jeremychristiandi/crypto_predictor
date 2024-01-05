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
from nbeats_block import NBeatsBlock

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
N_SPLIT = 0.8

# Hyperparameter utama penelitian
WINDOWS = ''
HORIZON = ''
BATCH_SIZE = 128

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

def predict_contents(df, ticker, windows, horizon):
    TICKER = ticker
    WINDOWS = windows
    HORIZON = horizon
    INPUT_SIZE = WINDOWS
    THETA_SIZE = INPUT_SIZE + HORIZON

    # Save path 
    SPLIT_STR = int(N_SPLIT * 100) 
    BASE_PATH = "saved_model"
    BASELINE_PATH = f"{BASE_PATH}_baseline/baseline_model_{TICKER}_{WINDOWS}_{HORIZON}_{BATCH_SIZE}_{SPLIT_STR}.keras"
    LSTM_PATH = f"{BASE_PATH}_lstm/lstm_model_{TICKER}_{WINDOWS}_{HORIZON}_{BATCH_SIZE}_{SPLIT_STR}.keras"
    NBEATS_PATH = f"{BASE_PATH}_nbeats/nbeats_model_{TICKER}_{WINDOWS}_{HORIZON}_{BATCH_SIZE}_{SPLIT_STR}.keras" 

    # 1. Menyisakan kolom date dan price
    df_date_price = pd.DataFrame(df["Close"]).rename(columns={"Close": "Price"})

    # 2. Forward filling: mengisi row (price) yang kosong dengan row sebelumnya
    if df_date_price["Price"].isnull().sum(axis=0) > 0:
        df_date_price.ffill(axis=0)

    plot_preview_prices(df_date_price)

    # Memisahkan data date dan price ke array masing-masing
    timesteps = df_date_price.index.to_numpy()
    prices = df_date_price["Price"].to_numpy()

    # Memisahkan dataset ke dalam bentuk data train dan test
    # Persentase splitting ditentukan oleh variabel N_SPLIT
    split_len = int(len(prices) * N_SPLIT)

    # X train dan test berisi data timesteps
    # y train dan test berisi data harga crypto
    X_train, X_test = timesteps[:split_len], timesteps[split_len:]
    y_train, y_test = prices[:split_len], prices[split_len:]

    df_prices = pd.DataFrame(df["Close"]).rename(columns={'Close' : 'Price'})
    crypto_prices = df_prices["Price"].to_numpy()

    all_windows, all_horizons = create_windows_horizons(prices, window=WINDOWS, horizon=HORIZON)
    windows_train, windows_test, horizons_train, horizons_test = create_dataset_splits(all_windows, all_horizons)
    
    # Naive Forecast Dropdown
    # with st.expander("Naive Model Prediction"):
    #     naive_res = y_test[:-1]
    #     visualize_naive(timesteps=X_test,
    #                    data=naive_res,
    #                    data_actual=y_test)

    # LSTM Dropdown
    with st.expander("LSTM Model Prediction"):
        lstm_pred, lstm_res = create_lstm_prediction(windows_train, windows_test, horizons_train, horizons_test)

        visualize_pred_actual(timesteps=X_test[-len(windows_test):],
                    data=lstm_pred, data_actual=np.squeeze(horizons_test))

    # with st.expander("NBeats Model Prediction"):
    #     with st_lottie_spinner(lottie_json, height=100):
    #         nbeats_pred, nbeats_res = create_nbeats_model(df_prices)

    #         nbeats_pred = nbeats_pred.numpy()
    #         visualize_pred_actual(timesteps=X_test[-len(windows_test):],
    #                         data=nbeats_pred, data_actual=np.squeeze(horizons_test))
    
    # with st.expander("Model Error Metrics"):
    #     visualize_errors(lstm_res, nbeats_res)

    # with st.expander("Models Error Metrics"):
    #     with st_lottie_spinner(lottie_json, height=100):
    #         if lstm_res != None and nbeats_res != None:
    #             visualize_errors(lstm_res, nbeats_res)
      

def plot_preview_prices(df_prices):
    with st.expander("Price Visualization"):
        st.line_chart(df_prices, x="Date", y="Price")

def visualize_naive(timesteps, data, data_actual):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timesteps, y=data_actual, name="Actual", marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=timesteps[1:], y=data, name="Prediction", marker=dict(color='red')))

    st.write(fig)

def visualize_pred_actual(timesteps, data, data_actual, start=0, end=None):
    data_timesteps = timesteps[start:end]
    pred_prices = data[start:end]
    actual_prices = data_actual[start:end]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_timesteps, y=pred_prices, name="Prediction",  marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data_timesteps, y=actual_prices, name="Actual", marker=dict(color='red')))

    st.write(fig)

def create_windowed_arr(data, horizon=HORIZON):
    return data[:, :-horizon], data[:, -horizon:]

def create_windows_horizons(data, window=WINDOWS, horizon=HORIZON):
    window_step = np.expand_dims(np.arange(window + horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(data) - (window + horizon - 1)), axis=0).T
    window_array = data[window_indexes]
    windows, horizons = create_windowed_arr(window_array, horizon=HORIZON)

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
    if len(y_pred) > 0:
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)

    return {
        "RMSE": rmse.numpy(),
        "MAPE": mape.numpy()
    }

def save_model_checkpoint(model_name, save_path="model_checkpoints"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), monitor="val_loss", verbose=0, save_best_only=True)

def create_dataset_splits(windows, horizons, train_size=0.8):
    split_len = int(len(windows) * train_size)
    windows_train = windows[:split_len]
    horizons_train = horizons[:split_len]
    windows_test = windows[split_len:]
    horizons_test = horizons[split_len:]

    return windows_train, windows_test, horizons_train, horizons_test

def create_prediction(model, test_data):
    pred_result = model.predict(test_data)
    return tf.squeeze(pred_result)

def visualize_errors(data_1, data_2):
    df_1 = pd.DataFrame(data_1, index=["LSTM"])
    df_2 = pd.DataFrame(data_2, index=["NBEATS"])
    data = pd.concat([df_1, df_2], axis=0)
    st.write(data)

def create_lstm_prediction(windows_train, windows_test, horizons_train, horizons_test):
    if os.path.exists(LSTM_PATH):
            lstm_model = load_model(LSTM_PATH, safe_mode=False)
    else:
        with st_lottie_spinner(lottie_json, height=100):
            tf.random.set_seed(42)

            # 1. Membangun Model LSTM 
            input_layers = layers.Input(shape=WINDOWS)
            x = layers.Lambda(lambda x : tf.expand_dims(x, axis=1))(input_layers)
            x = layers.LSTM(128, activation="relu")(x)
            output = layers.Dense(HORIZON)(x)
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

    lstm_pred = create_prediction(lstm_model, windows_test)
    lstm_res = create_error_metrics(y_actual=tf.squeeze(horizons_test),
                                    y_pred=lstm_pred) 

    return lstm_pred, lstm_res

def create_nbeats_model(df_prices):
    crypto_prices_temp = df_prices.copy()
    for i in range(WINDOWS):
        crypto_prices_temp[f"Price+{i+1}"] = crypto_prices_temp["Price"].shift(periods=i+1)
    
    X = crypto_prices_temp.dropna().drop("Price", axis=1)
    y = crypto_prices_temp.dropna()["Price"]

    split_percentage = 0.8
    split_len = int(len(X) * split_percentage)
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

    LAYERS = 4
    STACKS = 30

    # WINDOW = 7
    # HORIZON = 1

    INPUT_SIZE = WINDOWS * HORIZON
    THETA_SIZE = INPUT_SIZE + HORIZON

    # Pembuatan Model
    tf.random.set_seed(42)

    block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                theta_size=THETA_SIZE,
                                horizon=HORIZON,
                                n_neurons=N_NEURONS,
                                n_layers=LAYERS,
                                name="basic_block")
    
    stack_input = tf.keras.layers.Input(shape=(INPUT_SIZE), name="stack_input")

    residuals, forecast = block_layer(stack_input)

    for (i, j) in enumerate(range(STACKS - 1)):
        backcast, block_forecast = NBeatsBlock(input_size=INPUT_SIZE,
                                                theta_size=THETA_SIZE,
                                                horizon=HORIZON,
                                                n_neurons=N_NEURONS,
                                                n_layers=LAYERS,
                                                name=f"NBeatsBlock-{i}")(residuals)
        
        residuals = tf.keras.layers.subtract([residuals, backcast], name=f"Subtract-{i}")
        forecast = tf.keras.layers.add([forecast, block_forecast], name=f"Add-{i}")

    nbeats_model = tf.keras.Model(inputs=stack_input, outputs=forecast, name="nbeats_model")

    nbeats_model.compile(loss="mae",
                            optimizer=tf.keras.optimizers.Adam())
    
    nbeats_model.fit(train_dataset,
                        epochs=N_EPOCHS,
                        validation_data=test_dataset,
                        verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                        patience=200,
                                                        restore_best_weights=True),
                                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                    patience=100,
                                                    verbose=1)])
    
    
    nbeats_model_pred = create_prediction(nbeats_model, test_dataset)
    nbeats_model_res = create_error_metrics(y_real = y_test,
                                            y_pred = nbeats_model_pred)
    
    return nbeats_model_pred, nbeats_model_res