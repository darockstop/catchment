import numpy as np
import my_toolkit as mltools
from sys import maxsize
from copy import deepcopy as cpy
from math import inf
import csv
import matplotlib.pyplot as plt
import os
from pdb import set_trace
import csv
import pandas as pd
from sklearn import preprocessing

ID = 0
W = 1
Z = 2
E = 3
W_CH = 4

SETO = 0
VERS = 1
VIRG = 2

T_MSE = 2
V_MSE = 3
T_MCE = 4
V_MCE = 5


def parse_data(csv_filename):
    with open('./{}.csv'.format(csv_filename), 'r') as csvfile:
        df = pd.read_csv(csvfile)
        # df.drop(['PMgranite', 'QNov2015', 'perWet', 'RiverD', 'Slope', 'NO3/Cl', 'NO3-N (mg/L)', 'Water', 'area'], axis=1)
    # set_trace()
    # inputs = df.iloc[:, 2:-2]  # TODO: changed
    targets = df.iloc[:, -2]

    inputs = pd.concat([df['PMgranite'], df['PMschiste']], axis=1)


    inputs = np.array(inputs)
    targets = np.array(targets)

    _, num_ftrs = inputs.shape

    to_keep = []
    for i in range(num_ftrs):
        col = inputs[:, i]
        if not np.isnan(col).any():  # if there are NOT any nan vals
            to_keep.append(i)

    return normalize(inputs[:, to_keep]), targets


def write_row(filepath, data):
    with open(filepath, 'a') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(data)


def normalize(matrix):
    return (matrix - matrix.min(0)) / matrix.ptp(0)


def create_graphs(f, ext, results):
    results = np.array(results)
    plt.plot(results[:, 1], results[:, T_MSE], '#d18759')
    plt.plot(results[:, 1], results[:, V_MSE], '#f2bc9b')  # lighter color
    plt.xlabel('# epochs')
    plt.ylabel('MSE')
    x1, x2, _, y2 = plt.axis()
    plt.axis((x1, x2, 0, y2))
    plt.savefig('./{}/mse{}.png'.format(f, ext), dpi=300)
    plt.clf()


def get_w(num):
    return np.random.normal(0, .5, num)


def predict(x, net):
    hidden, out_layer = net
    for layer in range(len(hidden)):  # calc output to end of hidden layer
        my_num_nodes = len(hidden[layer]) - 1  # nodes in this layer (don't want bias)
        z = cpy(x) if layer == 0 else [node[Z] for node in
                                       hidden[layer - 1]]  # get last layer's output inc bias (new input)
        for n_i in range(my_num_nodes):
            curr_node = hidden[layer][n_i]
            net = np.sum(np.multiply(z, curr_node[W]))
            curr_node[Z] = 1 / (1 + np.e ** (-net))
    z = [node[Z] for node in hidden[-1]]  # last hidden layer's output

    for n_i in range(len(out_layer)):  # calc output
        curr_node = out_layer[n_i]
        net = np.sum(np.multiply(z, curr_node[W]))
        curr_node[Z] = 1 / (1 + np.e ** (-net))
    return [node[Z] for node in out_layer]  # return output


def test_acc(test_set, ts, net, csvfile):
    num_samples, num_ftrs = test_set.shape
    test_set = np.hstack((test_set, np.ones((num_samples, 1))))  # add bias
    sum_sq_error = 0.
    for x, y in zip(test_set, ts):
        output = predict(x, net)
        if csvfile:
            csv.writer(csvfile).writerow(['', np.round(y, 3)[0], np.round(output, 3)[0], np.abs(np.round(y-output, 3))[0]])
        sq_errors = np.square(np.subtract(output, y))
        sum_sq_error += sum(sq_errors)
    return sum_sq_error / num_samples


def hidden_layer(num_hidden, num_nodes, num_ftrs):
    H = []
    id_num = 0
    for h_i in range(num_hidden):
        my_num_nodes = num_nodes[h_i]
        h = []
        for _ in range(my_num_nodes):
            num_wts = num_ftrs + 1 if h_i == 0 else len(
                H[h_i - 1])  # if not first layer, num_wts is num nodes in prev layer
            h.append(["n{}".format(id_num), get_w(num_wts), 0, 0, np.zeros(num_wts)])  # add node
            id_num += 1
        else:
            num_wts = 0

        h.append(["b{}".format(id_num), 0, 1, 0, np.zeros(num_wts)])  # add bias node
        id_num += 1
        H.append(h)
    return np.array(H, dtype=object)


def output_layer(num_wts, num_trgs):
    layer = []
    for i in range(num_trgs):
        layer.append(['z{}'.format(i), get_w(num_wts), 0, 0, np.zeros(num_wts)])
    return np.array(layer, dtype=object)


def train(X, y, f, ext, num_nodes, l_r=1., m=0., max_no_change=10, frac_vs=.2, max_epochs=maxsize,
          prec=6):
    # print('TRAINING --- num_nodes={} l_r={} m={} max_no_change={}'.format(num_nodes, l_r, m, max_no_change))

    num_samples, num_ftrs = X.shape
    num_trgs = y.shape[1]
    num_hidden = len(num_nodes)  # num_nodes is a list of layers (lists)

    num_vs = round(num_samples * frac_vs)
    X_train = X[:num_samples - num_vs, :]
    X_val = X[num_samples - num_vs:, :]
    y_train = y[:num_samples - num_vs, :]
    y_val = y[num_samples - num_vs:, :]

    hidden = hidden_layer(num_hidden, num_nodes, num_ftrs)
    out_layer = output_layer(num_wts=len(hidden[-1]), num_trgs=num_trgs)
    X_train = np.hstack((X_train, np.ones((num_samples - num_vs, 1))))  # add bias column to train

    results = []
    best_error = inf  # for comparison, just using sum squared error
    best_sol = []
    no_change = 0
    num_epochs = 0
    best_train_MSE = 0

    for j in range(max_epochs):
        num_epochs += 1
        np.random.seed()  # change it up
        X_train, y_train = mltools.shuffle_rows(X_train, y_train)  # shuffle before starting new epoch
        train_squared_error = 0.
        for i in range(len(X_train)):  # one epoch
            x, t = X_train[i], y_train[i]  # curr input w bias, target

            for layer in range(num_hidden):  # calc output to end of hidden layer
                my_num_nodes = num_nodes[layer]  # nodes in this layer (don't want bias)
                z = cpy(x) if layer == 0 else [node[Z] for node in hidden[layer - 1]]  # get last layer's output inc bias (new input)
                for n_i in range(my_num_nodes):
                    curr_node = hidden[layer][n_i]
                    net = np.sum(np.multiply(z, curr_node[W]))
                    curr_node[Z] = 1 / (1 + np.e ** (-net))
            z = [node[Z] for node in hidden[-1]]  # last hidden layer's output

            for n_i in range(len(out_layer)):  # calc output and error of output layer
                curr_node = out_layer[n_i]
                curr_t = t[n_i]
                net = np.sum(np.multiply(z, curr_node[W]))
                curr_node[Z] = 1 / (1 + np.e ** (-net))
                curr_node[E] = (curr_t - curr_node[Z]) * curr_node[Z] * (1 - curr_node[Z])
            final_output = [node[Z] for node in out_layer]  # get final output
            sq_errors = np.square(np.subtract(final_output, t))  # calc errors
            train_squared_error += sum(sq_errors)

            for layer in reversed(range(num_hidden)):  # backprop error
                my_num_nodes = num_nodes[layer]
                fwd_layer = out_layer if layer == num_hidden - 1 else hidden[layer + 1][:-1]
                fwd_e = np.array([node[E] for node in fwd_layer]).reshape(len(fwd_layer), 1)

                for n_i in range(my_num_nodes):
                    curr_node = hidden[layer][n_i]
                    fwd_wts = [node[W] for node in fwd_layer]
                    curr_node[E] = np.sum(np.multiply(fwd_wts, fwd_e)) * curr_node[Z] * (1 - curr_node[Z])

            for layer in range(num_hidden):  # change weights through hidden layer
                my_num_nodes = num_nodes[layer]
                z = cpy(x) if layer == 0 else np.array(
                    [node[Z] for node in hidden[layer - 1]])  # get previous layer's output
                for n_i in range(my_num_nodes):
                    curr_node = hidden[layer][n_i]
                    curr_node[W_CH] = l_r * curr_node[E] * z + m * curr_node[W_CH]
                    curr_node[W] += curr_node[W_CH]

            z = np.array([node[Z] for node in hidden[-1]])  # change weights going to output layer
            for out_node in out_layer:  # change to index?
                out_node[W_CH] = l_r * out_node[E] * z + m * out_node[W_CH]
                out_node[W] += out_node[W_CH]

        # DONE WITH TRAINING EPOCH
        train_MSE = train_squared_error / num_samples

        vs_MSE = test_acc(X_val, y_val, (hidden, out_layer), None)

        results.append([0, num_epochs, round(train_MSE, prec), round(vs_MSE, prec)])

        if vs_MSE < best_error:  # compare with bssf
            best_error = cpy(vs_MSE)
            best_sol = (cpy(hidden), cpy(out_layer))
            best_train_MSE = cpy(train_MSE)
            no_change = 0
        else:
            no_change += 1 if num_epochs > 20 else 0

        if no_change > max_no_change:  # check if stop
            break

    # DONE WITH TRAINING
    # write_csv(f, ext, results)
    create_graphs(f, ext, results)

    return best_sol, num_epochs, best_error, best_train_MSE


def kfold(k, summ_file, dataset, num_nodes, alpha, m, no_change):
    np.random.seed()
    inputs, targets = parse_data(dataset)
    num_samples, num_ftrs = inputs.shape
    all_together = np.hstack((inputs, targets.reshape(num_samples, 1)))  # put inputs and targets together
    np.random.shuffle(all_together)  # shuffle

    kfolds = [[] for _ in range(k)]  # create k empty lists
    for i, row in enumerate(all_together):  # fill lists
        kfolds[i % k].append(row.tolist())
    kfolds = [np.array(x) for x in kfolds]  # now a list of 2d numpy arrays

    summ_file.writerow(['k', 'architecture', 'learning rate', 'm', '# epochs', 'train mse', 'val mse', 'test mse'])

    for i, fold in enumerate(kfolds):
        X_test, y_test = fold[:, :num_ftrs], fold[:, num_ftrs:]  # split inputs and outputs
        training_data = kfolds[:i] + kfolds[i + 1:]
        training_data = np.vstack(training_data)
        X_train, y_train = training_data[:, :num_ftrs], training_data[:, num_ftrs:]  # remainder for training

        network, num_epochs, vs_MSE, train_MSE = train(X_train, y_train, 'test', '_{}'.format(i), num_nodes, alpha, m, no_change,
                                                       max_epochs=500)

        with open('pred_{}.csv'.format(i), 'w') as pred_file:
            writer = csv.writer(pred_file)
            writer.writerow(['', 'trgt', 'pred', 'diff'])
            test_mse = test_acc(X_test, y_test, network, pred_file)

        print('# epochs={}, vsMSE={}, train_MSE={}'.format(round(num_epochs, 6), round(vs_MSE, 6), round(train_MSE, 6)))
        print('Test mse={}'.format(round(test_mse, 6)))
        summ_file.writerow([k, num_nodes, alpha, m, num_epochs, train_MSE, vs_MSE, test_mse])


def run(dataset, num_nodes, alpha, m, no_change):
    np.random.seed()

    all_inputs, all_targets = parse_data(dataset)
    all_targets = all_targets.reshape((all_targets.shape[0], 1))

    X_train, X_test, y_train, y_test = mltools.random_split(all_inputs, all_targets, test_fraction=.2)
    network, num_epochs, vs_MSE, train_MSE = train(X_train, y_train, 'test', '', num_nodes, alpha, m, no_change, max_epochs=500)

    with open('pred.csv', 'w') as pred_file:
        writer = csv.writer(pred_file)
        writer.writerow(['', 'trgt', 'pred', 'diff'])
        test_mse = test_acc(X_test, y_test, network, pred_file)

    print('Took {} epochs, vsMSE={}, train_MSE={}'.format(round(num_epochs, 6), round(vs_MSE, 6), round(train_MSE, 6)))
    print('Test mse: {}'.format(round(test_mse, 6)))

    scaler = preprocessing.StandardScaler()
    std_wts = scaler.fit_transform(network[0][0][0][W].reshape(-1, 1)).flatten()
    wt2 = network[1][0][W][0]
    return std_wts, wt2, test_mse

# with open('summary.csv', 'w') as f:
#     kfold(5, csv.writer(f), 'MLDatabase-NoSeasonal', [1], .1, 0, 10)

with open('weights.csv', 'w') as file:
    w = csv.writer(file)
    kfold(4, w, 'MLDatabase', [168], .01, .9, 20)

    # mses.append(to_add)

# print(np.round(mses, 4))
# print(np.average(mses))










# for repeat in range(3):
#     os.system('say "next round"')
#     test_values = [[1], [2], [4], [8], [16], [32], [64], [128], [256]]
#     folder = 'nodes/round{}'.format(repeat + 1)
#     do_vowel(folder, test_values)
# os.system('say "done"')

# # use PCA to reduce number of features
# all_inputs = StandardScaler().fit_transform(all_inputs)
# pca = PCA(.95)
# pca.fit(all_inputs)
# all_inputs = pca.transform(all_inputs)

# with open('./vowel/{}/summary.csv'.format(f), 'w') as csvf:
#     w = csv.writer(csvf)
#     w.writerow(['num nodes', '% Test Correct', 'Test MSE', 'VS MSE', 'Train MSE' '# epochs'])

# for test in test_vals:
#     ext = '{}'.format(test)
#     network, total_num_epochs, vs_MSE, train_MSE = train(X_train, y_train, 'vowel/{}'.format(f), ext,
#                                                          num_nodes=test,
#                                                          l_r=.1, m=.4, max_no_change=20, max_epochs=600)
#     test_mce, test_mse = test_acc(X_test, y_test, network)
#
#     with open('./vowel/{}/summary.csv'.format(f), 'a') as csvf:
#         w = csv.writer(csvf)
#         w.writerow([str(test), round(1 - test_mce, 6), round(test_mse, 6), round(vs_MSE, 6), round(train_MSE, 6),
#                     total_num_epochs])
#
#     print(1 - test_mce, test_mse, total_num_epochs)
#     print('\a')
