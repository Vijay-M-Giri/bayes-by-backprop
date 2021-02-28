import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

N = 500


def plot(model, x, y, x_test):
    y_hat = model(x_test)
    plt.plot(x, y, 'kx')
    plt.plot(x_test, y_hat.mean(), 'r')
    plt.show()


def plot_variational(model, x, y, x_test):
    y_hats = [model(x_test) for _ in range(100)]
    plt.plot(x, y, 'kx')
    # plt.plot(x_test, y_hats[0].mean(), 'r')
    for i in range(20):
        plt.plot(x_test, y_hats[i].mean(), 'r')
    # y_test_mean = sum([y_hat.mean() for y_hat in y_hats]) / len(y_hats)
    # plt.plot(x_test, y_test_mean, 'r')
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
    y = y[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return x, y, x_test


def neg_log_likelihood(y_true, p): return -p.log_prob(y_true)


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype),
                                                scale=tf.ones(n)),
                                     reinterpreted_batch_ndims=1)


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1))
    ])


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(1),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1.))
    ])
    return model


def build_variational_model():
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(
            10, activation='tanh', make_posterior_fn=posterior,
            make_prior_fn=prior, kl_weight=1/N, kl_use_exact=True),
        tfp.layers.DenseVariational(
            10, activation='tanh', make_posterior_fn=posterior,
            make_prior_fn=prior, kl_weight=1/N, kl_use_exact=True),
        tfp.layers.DenseVariational(
            1, activation='tanh', make_posterior_fn=posterior,
            make_prior_fn=prior, kl_weight=1/N, kl_use_exact=True),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1.))
    ])
    return model


def run(uncertainty=False):
    x, y, x_test = load_dataset(N)

    if uncertainty:
        model_func, plot_func = build_variational_model, plot_variational
    else:
        model_func, plot_func = build_model, plot

    model = model_func()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                  loss=neg_log_likelihood)
    model.fit(x, y, epochs=50)
    model.summary()
    model.save('./model')
    plot_func(model, x, y, x_test)


run(uncertainty=False)
