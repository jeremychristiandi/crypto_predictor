import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import yfinance as yf

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

# Global Variables
WINDOW = 7
HORIZON = 1

def predict_contents(df):
    df_data = pd.DataFrame(df[["Date","Close"]]).rename(columns={'Close' : 'Price'})
    plot_preview_prices(df_data, "Bitcoin")

    with st.expander("Train and Test Data"):
        st.header("Train and Test Split")
        st.write(f"Split Percentage **{80}%** *train* - **{20}%** *test*")
        timesteps = df_data["Date"].to_numpy()
        prices = df_data["Price"].to_numpy()

        split_percentage = 0.8
        split_len = int(len(prices) * split_percentage)

        X_train, y_train = timesteps[:split_len], prices[:split_len]
        X_test, y_test = timesteps[split_len:], prices[split_len:]

        st.header("Train Data")
        visualize_data(X_train, y_train)
        st.header("Test Data")
        visualize_data(X_test, y_test, color="#FFF239")

    df_prices = pd.DataFrame(df["Close"]).rename(columns={'Close' : 'Price'})
    crypto_prices = df_prices["Price"].to_numpy()

    all_windows, all_horizons = create_windows_horizons(crypto_prices, window=WINDOW, horizon=HORIZON)
    windows_train, windows_test, horizons_train, horizons_test = create_dataset_splits(all_windows, all_horizons)

    # with st.expander("LSTM Model Prediction"):
    #     with st_lottie_spinner(lottie_json, height=100):
    #         lstm_pred, lstm_res = create_lstm_model(windows_train, windows_test, horizons_train, horizons_test)
    #         st.write("LSTM Prediction:")
    #         st.write(lstm_res)
    #         visualize_data(timesteps=X_test[-len(windows_test):],
    #                     data=lstm_pred)

    with st.expander("NBeats Model Prediction"):
        with st_lottie_spinner(lottie_json, height=100):
            nbeats_pred, nbeats_res = create_nbeats_model(df_prices)
            st.write("NBeats Prediction:")
            st.write(nbeats_res)
            # visualize_data(timesteps=X_test[-len(windows_test):],

            st.write("Nbeats",nbeats_pred)
            st.write("Horizons",horizons_test)
            visualize_data_2(timesteps=X_test[-len(windows_test):],
                            data=nbeats_pred, data_actual=horizons_test)
            

def plot_preview_prices(df_prices, currency):
    with st.expander("Price Visualization"):
        st.header(f"{currency} data")
        st.line_chart(df_prices, x="Date", y="Price")

def visualize_data(timesteps, data, start=0, end=None, color="#4895EF"):
    data_dict = {
        "timesteps": timesteps[start:end],
        "price": data[start:end]
    }
    data_pd = pd.DataFrame(data_dict)
    st.scatter_chart(data_pd, x="timesteps", y="price", color=[color])

def visualize_data_2(timesteps, data, data_actual, start=0, end=None, color="#4895EF"):
    data_timesteps = timesteps[start:end]
    pred_prices = data[start:end]
    actual_prices = data_actual[start:end]

    st.write(len(data_timesteps))
    st.write(len(pred_prices))
    st.write(len(data_timesteps))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_timesteps, y=pred_prices, name="Prediction"))
    fig.add_trace(go.Scatter(x=data_timesteps, y=actual_prices, name="Actual"))

    st.write(fig)

def create_windowed_arr(x, horizon=HORIZON):
    return x[:, :-horizon], x[:, -horizon:]

def create_windows_horizons(x, window=WINDOW, horizon=HORIZON):
    window_step = np.expand_dims(np.arange(window + horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window + horizon - 1)), axis=0).T
    window_array = x[window_indexes]
    windows, horizons = create_windowed_arr(window_array, horizon=HORIZON)

    return windows, horizons

def evaluate_mase(y_real, y_pred):
    mae = tf.reduce_mean(tf.abs(y_real - y_pred))
    mae_no_season = tf.reduce_mean(tf.abs(y_real[1:] - y_real[:-1]))
    return mae / mae_no_season

def generate_predictions(y_real, y_pred):
    y_real = tf.cast(y_real, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mae = tf.keras.metrics.mean_absolute_error(y_real, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_real, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_real, y_pred)
    mase = evaluate_mase(y_real, y_pred)
    results = {
        "mae": [mae.numpy()],
        "mse": [mse.numpy()],
        "rmse": [rmse.numpy()],
        "mape": [mape.numpy()],
        "mase": [mase.numpy()],
    }

    return pd.DataFrame(results)

def save_model_checkpoint(model_name, save_path="model_checkpoints"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), monitor="val_loss", verbose=0, save_best_only=True)

def create_dataset_splits(windows, horizons, train_size=0.8):
    split_len = int(len(windows) * train_size)
    windows_train = windows[:split_len]
    horizons_train = horizons[:split_len]
    windows_test = windows[split_len:]
    horizons_test = horizons[split_len:]

    return windows_train, windows_test, horizons_train, horizons_test

def model_predict(model, test_data):
    pred_result = model.predict(test_data)
    return tf.squeeze(pred_result)

def create_lstm_model(windows_train, windows_test, horizons_train, horizons_test):
    tf.random.set_seed(42)

    input_layers = tf.keras.layers.Input(shape=(WINDOW))
    x = tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=1))(input_layers)
    x = tf.keras.layers.LSTM(128, activation="relu")(x)
    output = tf.keras.layers.Dense(HORIZON)(x)
    lstm_model = tf.keras.Model(input_layers, outputs=output, name="lstm_model")

    lstm_model.compile(loss="MAE",
                       optimizer=tf.keras.optimizers.Adam())
    
    lstm_model.fit(windows_train,
                   horizons_train,
                   epochs=100,
                   verbose=0,
                   batch_size=128,
                   validation_data=(windows_test, horizons_test),
                   callbacks=[save_model_checkpoint(model_name=lstm_model.name)])

    lstm_model = tf.keras.models.load_model("model_checkpoints/lstm_model")
    lstm_pred = model_predict(lstm_model, windows_test)
    lstm_res = generate_predictions(y_real = tf.squeeze(horizons_test),
                                    y_pred = lstm_pred)
    return lstm_pred, lstm_res

def create_nbeats_model(df_prices):
    crypto_prices_temp = df_prices.copy()
    for i in range(WINDOW):
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

    BATCH_SIZE = 1024

    train_dataset = train_dataset_zipped.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset_zipped.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    N_LAYERS = 4
    N_STACKS = 30
    N_NEURONS = 512
    N_EPOCHS = 5000

    # WINDOW = 7
    # HORIZON = 1

    INPUT_SIZE = WINDOW * HORIZON
    THETA_SIZE = INPUT_SIZE + HORIZON

    # Pembuatan Model
    tf.random.set_seed(42)

    block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                theta_size=THETA_SIZE,
                                horizon=HORIZON,
                                n_neurons=N_NEURONS,
                                n_layers=N_LAYERS,
                                name="basic_block")
    
    stack_input = tf.keras.layers.Input(shape=(INPUT_SIZE), name="stack_input")

    residuals, forecast = block_layer(stack_input)

    for (i, j) in enumerate(range(N_STACKS - 1)):
        backcast, block_forecast = NBeatsBlock(input_size=INPUT_SIZE,
                                                theta_size=THETA_SIZE,
                                                horizon=HORIZON,
                                                n_neurons=N_NEURONS,
                                                n_layers=N_LAYERS,
                                                name=f"NBeatsBlock-{i}")(residuals)
        
        residuals = tf.keras.layers.subtract([residuals, backcast], name=f"Subtract-{i}")
        forecast = tf.keras.layers.add([forecast, block_forecast], name=f"Add-{i}")

    nbeats_model = tf.keras.Model(inputs=stack_input, outputs=forecast, name="nbeats_model")

    nbeats_model.compile(loss="mae",
                            optimizer=tf.keras.optimizers.Adam())
    
    nbeats_model.fit(train_dataset,
                        epochs=3,
                        validation_data=test_dataset,
                        verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                        patience=200,
                                                        restore_best_weights=True),
                                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                    patience=100,
                                                    verbose=1)])
    
    
    nbeats_model_pred = model_predict(nbeats_model, test_dataset)
    nbeats_model_res = generate_predictions(y_real = y_test,
                                            y_pred = nbeats_model_pred)
    
    return nbeats_model_pred, nbeats_model_res