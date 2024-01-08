import tensorflow as tf

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

# class NBeatsBlock(tf.keras.layers.Layer):
#     def __init__(self,
#                  input_size: int,
#                  theta_size: int,
#                  horizon: int,
#                  n_neurons: int,
#                  n_layers: int,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.input_size = input_size
#         self.theta_size = theta_size
#         self.horizon = horizon
#         self.n_neurons = n_neurons
#         self.n_layers = n_layers

#         # Hidden layer -> FC stack, terdiri dari 4 hidden layers
#         self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for i in range(n_layers)]
#         # Theta layer -> output dari FC layer
#         self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

#     def call(self, inputs):
#         x = inputs
#         for hidden_layer in self.hidden:
#             x = hidden_layer(x)
#         theta = self.theta_layer(x)

#         # Hasil output dari theta layer berupa backcast and forecast
#         backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
#         return backcast, forecast
