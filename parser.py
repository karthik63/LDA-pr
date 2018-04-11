import argparse

parser = argparse.ArgumentParser(description='process arguments for LDA')

parser.add_argument('--dataset_path', default='data_banknote_authentication.npy', help='path to dataset to be used')
parser.add_argument('--labels_path', default='data_banknote_authentication_labels.npy', help='path to labels if separate')
parser.add_argument('--dataset_type', default='npy', help="numpy array or txt file ? numpy or txt")
parser.add_argument('--combined', default='False', help='are training data and labels combined ?')
parser.add_argument('--visualise', default='True', help='visalise data ?')
parser.add_argument('--dimensions', default='2', type=int, help='number of dimensions to which to reduce data ?')
parser.add_argument('--n_classes', default='2', type=int, help='no of classes')


def get_parser():
    return parser