import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

class LDA:
    def __init__(self, args):

        self.X = args['X']
        self.Y = args['Y']
        self.n_classes = args['n_classes']
        self.dimensions_original = self.X.shape[1]
        self.dimensions = args['dimensions']
        self.class_data = None


    def perform_lda(self):

        self.class_data = self.segregate()

        scatter_b, scatter_w = self.scatter()

        s_w_inv_s_b = np.matmul(np.linalg.inv(scatter_w), scatter_b)

        eigen_values, eigen_vectors = np.linalg.eig(s_w_inv_s_b)

        eigen_values_sorted_indices = np.argsort(eigen_values)[::-1]

        eigen_values_sorted_indices = eigen_values_sorted_indices[0:self.dimensions]

        w = eigen_vectors[:,eigen_values_sorted_indices]

        X_t = np.transpose(self.X)

        X_reduced_transpose = np.matmul(np.transpose(w), X_t)

        return np.transpose(X_reduced_transpose)

    def scatter(self):

        centroids_class = []

        centroid_total = np.zeros(self.dimensions_original, dtype=np.float32)

        for i in range(self.n_classes):
            centroids_class.append(np.sum(self.class_data[i], axis=0) / self.class_data[i].shape[0])

            centroid_total += centroids_class[i]

        centroid_total /= self.n_classes

        s_b = np.zeros((self.dimensions_original, self.dimensions_original), dtype = np.float32)
        s_w = np.zeros((self.dimensions_original, self.dimensions_original), dtype = np.float32)

        print(centroids_class[0].shape,'aaa')
        print(centroid_total.shape,'aaa')

        for i in range(self.n_classes):
            s_b += np.outer(centroids_class[i] - centroid_total, centroids_class[i] - centroid_total)

        for i in range(self.n_classes):
            for j in range(self.class_data[i].shape[0]):

                print(i,j)

                s_w += np.outer(self.class_data[i][j] - centroids_class[i], self.class_data[i][j] - centroids_class[i])

        return s_b, s_w

    def segregate(self):

        class_data = []

        Y_sorted_indices = self.Y.argsort()

        X_sorted = self.X[Y_sorted_indices]
        Y_sorted = self.Y[Y_sorted_indices]


        k = 0

        for i in range(self.n_classes):

            position = np.searchsorted(Y_sorted, i, side='right')

            class_data.append(X_sorted[k:position])

            k = position

        return class_data
