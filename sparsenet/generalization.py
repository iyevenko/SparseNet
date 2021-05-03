from sparsenet.train import *



def truncate_loss(g, order):
    i = 0
    while i < len(order):
        if order[i].name == 'out':
            break
        i += 1

    j = len(order) - 1
    while j > i:
        node = order[j]
        if isinstance(node, Operator):
            del g.operators[node.name]
        if isinstance(node, Constant):
            del g.constants[node.name]
        if isinstance(node, Placeholder):
            del g.placeholders[node.name]
        j -= 1
    g.tail = g.operators['out']

def train_mnist(layer_widths, epochs, learning_rate, g=None, start=0, exclude=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train = np.float32(x_train) / 255
    # x_test = np.float32(x_test) / 255
    ds_shape = x_train.shape
    print(f'Loaded {ds_shape[0]} examples from Mnist')

    history = []
    if g is None:
        layer_widths = [ds_shape[1] * ds_shape[2]] + layer_widths + [9]
        kernel_init = NormalInitializer(0.01)
        bias_init = NormalInitializer(0.01)
        g = VectorizedDenseGraph(layer_widths, kernel_init, bias_init)


    label = Placeholder(name='label')
    loss = sparse_categorical_cross_entropy(label, g.tail)

    ordered_graph = topological_sort(g, loss)
    # plot(ordered_graph).render(view=True)

    for i in range(epochs):
        x = np.reshape(x_train[i+start], (1, -1))
        y = np.reshape(y_train[i+start], (-1))

        if exclude == None or y not in exclude:
            feed_dict = {f'in/0': x}
            feed_dict['label'] = y
            train_step(g, ordered_graph, feed_dict, history, learning_rate=learning_rate)

    # plt.title('Custom')
    # plt.plot(history)
    # plt.show()

    truncate_loss(g, ordered_graph)

    test_indices = [np.nonzero(np.equal(y_test, i))[0] for i in range(10)]
    nums = [x_test[idx] for idx in test_indices]

    return g, nums


def eval_on_ds(ds, examples, g, order):
    activations = {'add/0': [], 'add/1': [], 'add/2': []}

    for ex in ds[:examples]:
        x = np.reshape(ex, (1, -1))
        feed_dict = {f'in/0': x}
        out = forward_pass(order, feed_dict=feed_dict)

        # for name, op in g.operators.items():
        #     if name[:2] == 'a/' or name == 'out':
        #         val = op.value
        #         if name == 'out':
        #             val = tf.nn.softmax(val).numpy()
        #         val = val[0]
        #
        #         activations[name].append(val)

        for name, op in g.operators.items():
            if name[:4] == 'add/':
                val = op.value[0]
                if name[4]=='2':
                    val = tf.nn.softmax(val).numpy()

                activations[name].append(val)


    mean_acts = []

    for name, a in activations.items():
        acts = np.stack(a)
        mean_acts.append(np.mean(acts, axis=0))
        if name == 'add/2':
            plt.title(name)
            plt.imshow(acts.T, cmap='Reds')
            plt.colorbar()
            plt.show()
            plt.close()

    return mean_acts

if __name__ == '__main__':
    layers = [32, 32]
    epochs = 10000
    lr = 1e-3

    np.random.seed(0)
    g, nums = train_mnist(layers, epochs, lr, exclude=[9])
    order = topological_sort(g, g.operators['out'])

    mean_acts = eval_on_ds(nums[9], 50, g, order)

    for i in range(len(mean_acts)-1):
        # a = mean_acts[i]
        # a_next = mean_acts[i+1]
        # a_new = 5 * np.max(a_next)
        # mean_acts[i+1] = np.concatenate([mean_acts[i+1], np.array([a_new])], axis=0)
        # w_new = a * (a_new / np.dot(a, a))
        # b_new = np.array([0])

        # w_old = g.variables[f'w/{i+1}'].value
        # g.variables[f'w/{i+1}'].value = np.concatenate([w_old, w_new[np.newaxis,:].T], axis=1)
        # b_old = g.variables[f'b/{i+1}'].value
        # g.variables[f'b/{i+1}'].value = np.concatenate([b_old, b_new], axis=0)
        #
        # if i != len(mean_acts) - 2:
        #     w_next_old = g.variables[f'w/{i+2}'].value
        #     g.variables[f'w/{i+2}'].value = np.concatenate([w_next_old, np.zeros_like(w_next_old[0:1,:])], axis=0)

        # Pad extra dimension to output of matmul
        old_w = g.variables[f'w/{i+1}'].value
        g.variables[f'w/{i+1}'].value = np.pad(old_w, ((0, 0), (0, 1)), 'constant', constant_values=0.01)
        # Create or pad an existing gradient mask to only update new weights
        old_mask = np.zeros_like(old_w) if g.variables[f'w/{i+1}'].grad_mask is None else g.variables[f'w/{i+1}'].grad_mask
        g.variables[f'w/{i+1}'].grad_mask = np.pad(old_mask, ((0, 0), (0, 1)), 'constant', constant_values=1)

        # Pad extra dimension to bias
        old_b = g.variables[f'b/{i+1}'].value
        g.variables[f'b/{i+1}'].value = np.pad(old_b, (0, 1), 'constant', constant_values=0.01)
        # Create or pad an existing gradient mask to only update new biases
        old_mask = np.zeros_like(old_b) if g.variables[f'b/{i+1}'].grad_mask is None else g.variables[f'b/{i+1}'].grad_mask
        g.variables[f'b/{i+1}'].grad_mask = np.pad(old_mask, (0, 1), 'constant', constant_values=1)

        if i != len(mean_acts) - 2:
            # Pad extra dimension to input of next matmul
            old_w = g.variables[f'w/{i+2}'].value
            g.variables[f'w/{i+2}'].value = np.pad(old_w, ((0, 1), (0, 0)), 'constant', constant_values=0)
            # Create or pad an existing gradient mask to only update new weights
            # old_mask = np.zeros_like(old_w) if g.variables[f'w/{i+2}'].grad_mask is None else g.variables[f'w/{i+2}'].grad_mask
            # g.variables[f'w/{i+2}'].grad_mask = np.pad(old_mask, ((0, 1), (0, 0)), 'constant', constant_values=1)

    # for _, var in g.variables.items():
    #     var.grad_mask = None

    train_mnist(layers, 10000, 1e-3, g, start=epochs)
    eval_on_ds(nums[4], 50, g, order)
    eval_on_ds(nums[9], 50, g, order)

    i = -1
    data = []
    a = mean_acts[0]
    for name, var in g.variables.items():
        if name[0] == 'w':
            if i >= 0:
                w = var.value
                a = a @ w + g.variables[f'b/{name[-1]}'].value
                data.append((name, w[:,-1], a))
                # print(f'{name}: {a@w}')
            i+=1

    for name, w, a in data:
        x = np.arange(a.size)
        width = 0.05

        plt.title(name)
        plt.bar(x+width/2, w, width, label='New Neurons')
        plt.bar(x-width/2, a, width, label='Activations')
        plt.show()
        plt.close()


