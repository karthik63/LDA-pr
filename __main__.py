import numpy as np
import matplotlib.pyplot as plt
import argparse as py
import parser
import converter
import lda
import time


def visualize(X_reduced, Y, n_classes):
    class_data = []

    Y_sorted_indices = Y.argsort()

    X_sorted = X_reduced[Y_sorted_indices]
    Y_sorted = Y[Y_sorted_indices]

    k = 0

    for i in range(n_classes):
        position = np.searchsorted(Y_sorted, i, side='right')

        class_data.append(X_sorted[k:position])

        k = position

    colors = ['r', 'b']

    for i in range(n_classes):

        plt.scatter(class_data[i][:,0], class_data[i][:,1], color=colors[i])

    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('LDA')
    plt.legend()
    plt.show()

def main():

    args = parser.get_parser().parse_args()

    dataset_path = args.dataset_path
    labels_path = args.labels_path
    dataset_type = args.dataset_type
    n_classes = args.n_classes
    dimensions = args.dimensions

    X = np.zeros(1)
    Y = np.zeros(1)

    visualise = False

    if args.visualise == 'True':
        visualise = True

    labels_path = args.labels_path

    combined = False

    if args.combined == 'True':
        combined = True

    if dataset_type == 'txt':

        if combined:
            converter.convert(dataset_path, combined)

        else:
            converter.convert(dataset_path, combined, labels_path=labels_path)

        dataset_name = dataset_path.strip().split('/')[-1][:-4]

        X = np.load('datasets_npy/' + dataset_name + '.npy')
        Y = np.load('datasets_npy/' + dataset_name + '_labels' + '.npy')

    elif dataset_type == 'npy':

        if combined:

            data = np.load(dataset_path)

            X = data[:, 0:-1]

            Y = data[:, -1]

        else:

            X = np.load(dataset_path)

            Y = np.load(labels_path)


    print('main')
    print(X.shape)
    print(Y.shape)
    print('main')

    myLDA = lda.LDA({'X':X, 'Y':Y, 'n_classes':n_classes, 'dimensions':dimensions})

    X_reduced = myLDA.perform_lda()

    np.save('aay', X_reduced)

    visualize(X_reduced, Y, n_classes)



time = time.time();
main()
print(time.time() - time)