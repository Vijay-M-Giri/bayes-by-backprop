import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def plot_wo_tfp(model, x, y, x_test):
    y_hat = model(x_test)
    plt.plot(x, y, 'kx')
    # plt.plot(x[:150], y[:150], 'kx')
    plt.plot(x_test, y_hat, 'r')
    plt.show()


def plot(model, x, y, x_test):
    y_hats = [model(x_test) for _ in range(100)]
    plt.plot(x[:150], y[:150], 'kx')
    # plt.plot(x_test, y_hats[0].mean(), 'r')
    y_test_mean = sum([y_hat.mean() for y_hat in y_hats]) / len(y_hats)
    plt.plot(x_test, y_test_mean, 'r')
    plt.show()


def load_dataset(n=150, n_test=150):
    low = 0.0
    high = 0.5
    rng = np.random.default_rng()
    x = rng.random(n) * (high - low) + low
    eps = rng.normal(loc=0.0, scale=0.02, size=n)
    y = x + 0.3 * np.sin(2 * np.pi * (x + eps)) + \
        0.3 * np.sin(4 * np.pi * (x + eps)) + eps
    x_test = np.linspace(low - 0.4, high + 0.8, n_test)
    x = x[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return x, y, x_test


def neg_log_likelihood(y_true, p): return -p.log_prob(y_true)


def mean_square_error(y_true, y_pred): return (y_pred - y_true) ** 2


def build_model_wo_tfp():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    return model


def build_model():
    model = tf.keras.Sequential([
        tfp.layers.DenseReparameterization(10, activation='tanh'),
        tfp.layers.DenseReparameterization(10, activation='tanh'),
        tfp.layers.DenseReparameterization(1, activation='tanh'),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1.))
    ])
    return model


def run(uncertainty=False):
    x, y, x_test = load_dataset(500)

    if uncertainty:
        model_func, loss_func, plot_func = \
            build_model, neg_log_likelihood,  plot
    else:
        model_func, loss_func, plot_func = \
            build_model_wo_tfp, mean_square_error,  plot_wo_tfp

    model = model_func()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                  loss=loss_func)
    model.fit(x, y, epochs=50)
    print(sum([np.prod(w.numpy().shape) for w in model.weights]))
    model.summary()
    model.save('./model')
    plot_func(model, x, y, x_test)


run(uncertainty=True)
