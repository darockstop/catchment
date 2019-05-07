import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_arff(file_path):
    """
    Loads arff file with scipy, returns a 2-tuple with first a numpy array of tuples and
    second, a list of 3-tuple with first, the var name, second, the var type ('nominal' or
    'numeric' and third a list of possible values nominal vars can take on (empty list if numeric)
    """
    data, meta = arff.loadarff(file_path)

    # consequent code creates info dictionary which I find useful
    var_names = meta.names()  # get variable names and types from scipy meta object
    var_types = meta.types()
    lines = str(meta).splitlines()[1:]  # get strings (which has the values each nominal variable can take)
    var_info = []
    for i in range(len(var_names)):  # index through variables
        if var_types[i] == 'nominal':  # if nominal, I want the values it can take
            values = lines[i][lines[i].find('(')+1:lines[i].rfind(')')].replace("'", "").split(",")  # just trust this
            var_info.append((var_names[i], 'nominal', values))
        else:
            var_info.append((var_names[i], 'numeric', []))  # I chose an empty list for nominal values if numeric

    return data, var_info

def normalize_matrix(matrix):
    """ Normalizes a numpy matrix so col vals are between 0 and 1."""
    return (matrix - matrix.min(0)) / matrix.ptp(0)
#    return normalize(matrix, axis=0, norm='max')

def shuffle_rows(X, y):
    """ Intended to shuffle the rows of a 2d numpy array """
    num_samples, num_ftrs = X.shape
    all_together = np.hstack((X, y))
    np.random.shuffle(all_together)
    return all_together[:,:num_ftrs], all_together[:, num_ftrs:]

def shuffle_all(X, y, vs):
    """ Intended to shuffle the rows of a 2d numpy array """
    num_samples, num_ftrs = X.shape
    all_together = np.hstack((X, y, vs))
    np.random.shuffle(all_together)
    return all_together[:,:num_ftrs], all_together[:, num_ftrs:-1], all_together[:, -1].reshape((vs.shape[0], 1))

def static_split(X, y, test_fraction=0.25, seed=None):
    """ Split data into training and testing set. First part always used for training, last part for testing. """
    return train_test_split(X, y, test_size=test_fraction, random_state=seed, shuffle=False)

def random_split(X, y, test_fraction=0.25, seed=None):
    """ Split data randomly into training and testing set """
    return train_test_split(X, y, test_size=test_fraction, random_state=seed, shuffle=True)

def evaluation_score(targets, predictions):
#     if report:
#         print(classification_report(targets, predictions))
    return accuracy_score(targets, predictions)


def k_fold_cross(X, y, k=10, train_func=None, args=None):
    """
    Perform k fold cross validation on a training function. If no function, simply returns a list of
    2d numpy matrices (of k length) that you can then manually use. If a function is provided then
    returns the average accuracy score.
    :X: a numpy 2d array
    :y: a numpy 2d array
    :k: int, number of splits
    :train_func: the function that will do the training
    :args: the arguments for the function (assumes that it takes train and test as first two arguments)
    """
    sum_samples, num_ftrs = X.shape
    all_together = np.hstack((X, y))  # put inputs and targets together
    np.random.shuffle(all_together)  # shuffle

    kfolds = [[] for _ in range(k)]  # create k empty lists
    for i, row in enumerate(all_together): # fill lists
        kfolds[i%k].append(row.tolist())

    kfolds = [np.array(x) for x in kfolds]  # now a list of 2d numpy arrays

    if train_func is None:
        return kfolds

    scores = []
    for i, fold in enumerate(kfolds):
        X_test, y_test = fold[:,:num_ftrs], fold[:, num_ftrs:]  # split inputs and outputs
        training_data = kfolds[:i] + kfolds[i+1:]
        training_data = np.vstack(training_data)
        X_train, y_train = training_data[:,:num_ftrs], training_data[:, num_ftrs:]  # remainder for training

        args.insert(0, X_train)  # put data in args
        args.insert(1, y_train)
        y_pred = train_func(args)  # get predictions from training

        scores.append(evaluation_score(y_test, y_pred))

    return sum(scores)/len(scores)  # return mean score


def print_table(data, padding=2, logfile=None):
    columns = np.asarray([data[:, c] for c in range(len(data[0]))])
    c_widths = []
    for col in columns:
        c_widths.append(max([len(datum) for datum in col]) + padding)

    for row in data:
        if logfile is None:
            print("".join(word.ljust(c_widths[i]) for i, word in enumerate(row)))
        else:
            with open(logfile, "a") as opened:
                opened.write("".join(word.ljust(c_widths[i]) for i, word in enumerate(row)))
                opened.write("\n")
    if logfile is None:
        print("{} updates before finishing".format(len(data)-1))
    else:
        with open(logfile, "a") as opened:
            opened.write("{} updates before finishing\n".format(len(data)-1))

def test_accuracy(X, y, weights):
    """ For single perceptron only. """
    X = np.hstack((X, np.ones((len(X), 1))))  # add bias column
    weighted = np.multiply(X, weights)
    thetas = np.sum(weighted, axis=1)
    outputs = np.where(thetas>0, 1, 0)
    return 1 - len([i for i in filter(lambda x: x[0] != x[1], zip(y, outputs))]) / len(X)


# import numpy as np
# inputs = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
# targets = np.array([1, 1, 1, 1, 0, 0, 0, 0])
# k = 2
