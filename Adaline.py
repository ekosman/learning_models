import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from os import path, mkdir
from itertools import product


SEED = 42
output_dir = r'graphs'
EPOCHS = 400
LEARNING_RATE = 0.00001

if not path.exists(output_dir):
    mkdir(output_dir)


class Adaline:
    def __init__(self, input_dim, lr, classes):
        """
        Initializes the classifier's weights
        :param input_dim: The dimension of the input
        :param lr: learning rate for the algorithm
        :param classes: classes\labels of the dataset
        """
        self.w = [np.random.uniform(-1, 1, (input_dim + 1, 1)) / np.square(input_dim) for i in range(len(classes))]
        self.lr = lr
        self.classes = classes

    @staticmethod
    def concat_ones(x):
        n = np.ones((x.shape[0], 1))
        return np.hstack((x, n))

    def print_training_log(self, verbose, train_x, train_y, val_x, val_y, epoch):
        if verbose == 0:
            score_train = self.score(train_x[:, :-1], train_y)

            score_val = None if val_x is None else self.score(val_x[:, :-1], val_y)
            print(f'Epoch {epoch}:     acc - {score_train}     val_acc - {score_val}')
            return score_train, score_val

    def fit(self, train_x, train_y, val_x=None, val_y=None, max_epochs=-1, target_acc=None, verbose=0):
        """
        :param train_x: train set features
        :param train_y: train set labels
        :param val_x: validation set features
        :param val_y: validation set labels
        :param max_epochs: maximum number of epoches
        :param target_acc: if max_epoch is not given, use this stopping criterion
        :param verbose: 0 - print logs (e.g. losses and accuracy)
                        1 - don't print logs
        """
        epoch = 1

        mappers = [np.vectorize(lambda x, c=c: 1 if x == c else -1) for c in self.classes]

        # creating labels for each 1-vs-all classifier
        orig_train_y = train_y
        train_y = np.array([mappers[i](train_y) for i in range(len(mappers))])

        # Adding another dimension for the bias term
        train_x = Adaline.concat_ones(train_x)
        if val_x is not None:
            val_x = Adaline.concat_ones(val_x)

        scores_train = []
        scores_val = []

        while True:
            training_order = np.random.permutation(range(train_x.shape[0]))
            for i_sample in training_order:
                for i, (w, c_train_y) in enumerate(zip(self.w, train_y)):
                    x_, y_ = train_x[i_sample].reshape(-1, 1), c_train_y[i_sample]
                    y_pred = self.forward(i, x_)[0][0]
                    if abs(y_pred) > 1000:
                        return

                    self.w[i] += self.lr * (y_ - y_pred) * x_

            score_train, score_val = self.print_training_log(verbose, train_x, orig_train_y, val_x, val_y, epoch)
            scores_train.append(score_train)
            scores_val.append(score_val)

            epoch += 1
            if max_epochs != -1 and epoch >= max_epochs:
                break
            elif max_epochs == -1 and score_val >= target_acc:
                break

        return scores_train, scores_val

    def forward(self, i, x):
        return self.w[i].T @ x

    def predict_prob(self, x):
        return [self.forward(i, x) for i in range(len(self.w))]

    def predict(self, x):
        return np.argmax(np.ravel(self.predict_prob(x)))

    def score(self, x, y):
        x = Adaline.concat_ones(x)
        preds_val = np.array([self.predict(x[i].reshape(-1, 1)) for i in range(x.shape[0])])
        preds_val = self.classes[preds_val]
        return np.mean(preds_val == y)


def make_perceptron_friendly_classification():
    x, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=1.0,
        scale=1,
        random_state=SEED
    )

    y = 2.0 * y - 1.0

    return x, y


def make_adaline_friendly_classification():
    x, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=2,
        flip_y=0.10,
        class_sep=0.25,
        scale=5,
        random_state=SEED
    )

    y = 2.0 * y - 1.0

    return x, y


def visualize_data(name, x, y, w_perc=None, w_ada=None):
    one_indices = np.where(y == 1)[0]
    minus_one_indices = np.where(y == -1)[0]

    one_dots = np.array(x)[one_indices]
    minus_one_dots = np.array(x)[minus_one_indices]

    plt.figure()
    plt.scatter(one_dots[:, 0], one_dots[:, 1])
    plt.scatter(minus_one_dots[:, 0], minus_one_dots[:, 1])

    top = max([e[1] for e in x])
    bottom = min([e[1] for e in x])

    if w_perc is not None:
        if 2 == w_perc.shape[0]:
            c = 0.0
            a, b = w_perc
        else:
            c, a, b = w_perc

        plt.plot([-(b * top + c) / a, -(b * bottom + c) / a], [top, bottom], 'g', label='perceptron decision line')

    if w_ada is not None:
        if 2 == w_ada.shape[0]:
            c = 0.0
            a, b = w_ada
        else:
            c, a, b = w_ada
        plt.plot([-(b * top + c) / a, -(b * bottom + c) / a], [top, bottom], 'y', label='Adaline decision line')
    plt.legend()
    plt.title(name + ' dataset')
    plt.savefig(path.join(output_dir, 'data_visualization_{}_dataset.png'.format(name)))


def train_on_all_data_and_return_weights_dict(clf_type, x, y):
    if clf_type == 'perceptron':
        clf = Perceptron(
            max_iter=1000,
            fit_intercept=False)
        clf.fit(x, y)
        return np.hstack((np.squeeze(clf.intercept_), np.squeeze(clf.coef_)))
    elif clf_type == 'adaline':
        clf = Adaline(input_dim=2, lr=1e-4, classes=np.unique(y))
        scores_train, scores_val = clf.fit(x, y, max_epochs=1000)
        w = np.squeeze(clf.w)[0]
        b = w[-1]
        w = w[:-1]
        return np.hstack([b, w]), scores_train, scores_val


def plot_training_curve(scores_train, scores_val, name):
    plt.figure()
    plt.title(f'Learning accuracy of Adaline on {name} dataset')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(scores_train, label='train_acc')
    plt.plot(scores_val, label='val_acc')
    plt.legend()
    plt.savefig(path.join(output_dir, f'adaline_on_{name}.png'))
    plt.close()


def cmp_perceptron_and_adaline_special(x, y, test_name):
    perc_weights = train_on_all_data_and_return_weights_dict('perceptron', x, y)
    ada_weights, scores_train, scores_val = train_on_all_data_and_return_weights_dict('adaline', x, y)
    # plot_training_curve(scores_train, scores_val, test_name)
    visualize_data(test_name, x, y, w_perc=perc_weights, w_ada=ada_weights)


def main():
    cmp_perceptron_and_adaline_special(*make_perceptron_friendly_classification(), "perceptron friendly")
    cmp_perceptron_and_adaline_special(*make_adaline_friendly_classification(), "adaline friendly")


if __name__ == '__main__':
    main()
