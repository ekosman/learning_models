from os import path, mkdir
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

SEED = 42
output_dir = r'perceptron_graphs'
EPOCHS = 400
LEARNING_RATE = 0.00001

if not path.exists(output_dir):
    mkdir(output_dir)


class Perceptron:
    def __init__(self, input_dim, lr, classes):
        """
        Initializes the classifier's weights
        :param input_dim: The dimension of the input
        :param lr: learning rate for the algorithm
        :param classes: classes\labels of the dataset
        """
        self.w = np.random.uniform(-1, 1, (input_dim + 1, 1)) / np.square(input_dim)
        self.lr = lr
        self.classes = classes
        self.mapper = np.vectorize(lambda x: 1 if x == classes[0] else -1)
        self.back_mapper = np.vectorize(lambda x: classes[0] if x == 1 else classes[1])

    @staticmethod
    def concat_ones(x):
        n = np.ones((x.shape[1], 1))
        return np.vstack((x, n))

    def print_training_log(self, verbose, train_x, train_y, epoch):
        if verbose == 0:
            score_train = self.score(train_x[:, :-1], train_y)

            print(f'Epoch {epoch}:     acc - {score_train}')
            return score_train

    def fit(self, train_x, train_y, max_epochs=-1, target_acc=None, verbose=0):
        """
        :param train_x: train set features
        :param train_y: train set labels
        :param max_epochs: maximum number of epoches
        :param target_acc: if max_epoch is not given, use this stopping criterion
        :param verbose: 0 - print logs (e.g. losses and accuracy)
                        1 - don't print logs
        """
        epoch = 1

        # Adding another dimension for the bias term
        mapped_train_y = self.mapper(train_y)

        scores_train = []
        weights = []

        while True:
            training_order = np.random.permutation(range(train_x.shape[0]))
            changed = False
            for i_sample in training_order:
                x_, y_true = train_x[i_sample].copy().reshape(-1, 1), mapped_train_y[i_sample]
                y_pred = self.predict(x_)

                if y_pred != y_true:
                    changed = True
                    self.w += self.lr * y_true * Perceptron.concat_ones(x_)

            if not changed:
                break

            # score_train = self.print_training_log(verbose, train_x, mapped_train_y, epoch)
            # scores_train.append(score_train)

            epoch += 1
            weights.append(self.w.copy())
            if max_epochs != -1 and epoch >= max_epochs:
                break
            # elif max_epochs == -1 and score_train >= target_acc:
            #     break

        return scores_train, weights

    def forward(self, x):
        return self.w.T @ x

    def predict(self, x):
        return np.sign(self.forward(self.concat_ones(x)))

    def score(self, x, y):
        x = Perceptron.concat_ones(x)
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

    return x, y.astype(np.int)


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


def visualize_data(name, x, y, w, clf):
    one_indices = np.where(y == 1)[0]
    minus_one_indices = np.where(y == -1)[0]

    one_dots = np.array(x)[one_indices]
    minus_one_dots = np.array(x)[minus_one_indices]

    plt.figure()

    top = max([e[1] for e in x]) + 1
    right = max([e[0] for e in x]) + 1
    bottom = min([e[1] for e in x]) - 1
    left = min([e[0] for e in x]) - 1
    xx = np.arange(left, right, 0.01)

    if 2 == w.shape[0]:
        c = 0.0
        a, b = w
    else:
        c, a, b = w

    def f(z):
        return - (a * z + c) / b

    y0 = f(0) + 1

    yy = np.ravel([f(z) for z in xx])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(xx, yy, 'g')
    plt.scatter(one_dots[:, 0], one_dots[:, 1])
    plt.scatter(minus_one_dots[:, 0], minus_one_dots[:, 1])
    plt.title(name + ' dataset')
    plt.xlim((left, right))
    plt.ylim((bottom, top))
    plt.savefig(path.join(output_dir, 'data_visualization_{}_dataset.png'.format(name.replace(' ', '_'))))
    plt.show()


def convert_weights(w):
    w = np.squeeze(w)
    b = w[-1]
    w = w[:-1]
    return np.hstack([b, w])


def train_on_all_data_and_return_weights_dict(x, y):
    clf = Perceptron(input_dim=2, lr=3e-4, classes=np.unique(y))
    scores_train, weights = clf.fit(x, y, max_epochs=1000)
    return weights, scores_train, clf


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


def train_on_dataset(x, y, test_name):
    print(test_name)
    weights, scores_train, clf = train_on_all_data_and_return_weights_dict(x, y)
    # plot_training_curve(scores_train, scores_val, test_name)
    for i, w in enumerate(weights):
        visualize_data(test_name+'_'+str(i), x, y, w=convert_weights(w), clf=clf)


def main():
    train_on_dataset(*make_perceptron_friendly_classification(), "linearly separable")
    # train_on_dataset(*make_adaline_friendly_classification(), "linearly inseparable")


if __name__ == '__main__':
    main()
