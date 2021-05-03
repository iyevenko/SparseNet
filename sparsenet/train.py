import time

from sparsenet.initializers import *
from sparsenet.sort import *
from sparsenet.traverse import *

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_TF_model(layer_widths, epochs, learning_rate):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    for w in layer_widths:
        model.add(tf.keras.layers.Dense(w, activation='relu',
                                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                        bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))

    model.add(tf.keras.layers.Dense(10,
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    # history = model.fit(x_train, y_train, batch_size=1, epochs=1, steps_per_epoch=100)
    # print(history.history['loss'])
    #
    history = []
    for i in range(epochs):
        with tf.GradientTape() as tape:
            x = x_train[tf.newaxis, i]
            y_pred = model(x)
            y = y_train[tf.newaxis, i]
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred, from_logits=True)
            history.append(float(loss))
            # print(f'TF: {float(loss)}')

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # if i % 1000 == 0:
        #     print(f'Batch: {i}, Loss = {history[-1]}')
    # print(grads)

    # i = 0
    # for grad in grads:
    #     name = f'W_{i//2}' if i %2 == 0 else f'b_{i//2}'
    #     plt.title(name)
    #     plt.hist(grad.numpy().flatten(), density=True, bins=50)
    #     plt.savefig(f'grad_hists/tf/{name}.png')
    #     plt.close()
    #     i+= 1

    plt.title('TF')
    plt.plot(history)
    plt.show()
    plt.close()

def train_step(g, order, feed_dict, history, learning_rate):
    # y_pred -> numpy array of logits
    loss = forward_pass(order, feed_dict=feed_dict)
    # print(loss)
    history.append(loss)
    grads = backward_pass(order)
    var_grads = [(name, var.gradient) for name, var in g.variables.items()]
    # print(var_grads)

    #
    # i = 0
    # for var_grad in var_grads:
    #     i+= 1
    #     name, grad = var_grad
    #     plt.title(f'{name}')
    #     plt.hist(grad.flatten(), density=True, bins=50)
    #     plt.savefig(f'grad_hists/custom/{name}.png')
    #     plt.close()


    # grads_list = {}
    # while len(var_grads) != 0:
    #     prefix = var_grads[0][0][:3]
    #     layer_grads=[]
    #     i = 0
    #     while i < len(var_grads):
    #         name, grad = var_grads[i]
    #         if name[:3] == prefix:
    #             layer_grads.append(var_grads.pop(i))
    #             i -= 1
    #         i += 1
    #     layer_grads.sort(key=lambda tup: tup[0])
    #     grads_list[prefix] = ([g for n, g in layer_grads])
    # for grad in grads_list.items():
    #     print(grad)
    # -- Add Gradient Clipping here --

    g.apply_gradients(learning_rate=learning_rate)


def train_static_dense_mnist(layer_widths, epochs, learning_rate):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    ds_shape = x_train.shape
    print(f'Loaded {ds_shape[0]} examples from Mnist')
    layer_widths = [ds_shape[1] * ds_shape[2]] + layer_widths + [10]

    history = []
    # kernel_init = ConstantInitializer(0.01)
    kernel_init = NormalInitializer(0.01)
    # bias_init = ConstantInitializer(0)
    bias_init = NormalInitializer(0.01)
    with VectorizedDenseGraph(layer_widths, kernel_init, bias_init) as g:
        label = Placeholder(name='label')
        loss = sparse_categorical_cross_entropy(label, g.tail)

        ordered_graph = topological_sort(g, loss)
        # plot(ordered_graph).render(view=True)

        for i in range(epochs):
            x = np.reshape(x_train[i], (1, -1))
            y = np.reshape(y_train[i], (-1))
            # feed_dict = {f'in/{i}': x[i] for i in range(x.shape[0])}
            feed_dict = {f'in/0': x}
            feed_dict['label'] = y
            train_step(g, ordered_graph, feed_dict, history, learning_rate=learning_rate)
            if i % 10000 == 0:
                print(f'Batch: {i}, Loss = {history[-1]}')

    plt.plot(history)
    plt.show()


def train_static_dense_MPG(layer_widths):
    # --- TensorFlow Tutorial stuff --- #
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = np.array(train_features.pop('MPG'))
    test_labels = np.array(test_features.pop('MPG'))
    train_features = np.array(train_features)
    test_features = np.array(test_features)

    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(train_features)

    # --- TensorFlow Tutorial stuff --- #
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(), bias_initializer=tf.keras.initializers.RandomNormal()),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(), bias_initializer=tf.keras.initializers.RandomNormal()),
        tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(), bias_initializer=tf.keras.initializers.RandomNormal())
    ])
    # model.compile(optimizer=tf.keras.optimizers.SGD(0.5), loss='mean_squared_error')
    history = []
    epochs = 100
    learning_rate = 0.04
    optimizer=tf.keras.optimizers.SGD(learning_rate)


    start = time.time()
    for i in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = tf.squeeze(model(train_features))
            loss = tf.keras.losses.mean_absolute_error(tf.constant(train_labels),y_pred)
            # print(loss)
            history.append(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        # print([g.numpy() for g in grads])
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    end = time.time()
    print(f'Tensorflow time: {end-start}')
    plt.plot(history)
    plt.show()

    num_inputs = train_features.shape[-1]
    kernel_init = NormalInitializer(0.05)
    bias_init = NormalInitializer(0.05)

    with VectorizedDenseGraph([num_inputs, 64, 64, 1], kernel_init, bias_init) as g:
        mpg = Placeholder(name='mpg')
        loss = mean_absolute_error(mpg, g.tail)
        order = topological_sort(g, loss)
        plot(order).render(view=True)

        x = normalizer(train_features).numpy()
        # x = np.squeeze(x, 1)
        # y = mpg_normalizer(train_labels).numpy()
        y = np.expand_dims(train_labels, -1)
        history = []
        feed_dict = {f'in/0': x}
        # feed_dict = {f'in/{i}': x[:, i] for i in range(num_inputs)}
        feed_dict['mpg'] = y
        start = time.time()
        for i in range(epochs):
            train_step(g, order, feed_dict, history, learning_rate)
            # print(f'Epoch: {i}, Loss = {history[-1]}')
        end = time.time()
    print(f'SparseNet time: {end-start}')
    plt.plot(history)
    plt.show()


if __name__ == '__main__':
    # train_static_dense_MPG([16])
    epochs = 1000
    lr = 0.001
    train_static_dense_mnist([1024, 1024, 1024], epochs, lr)
    train_TF_model([1024, 1024, 1024], epochs, lr)