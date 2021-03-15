# atlas = 'cc200', model = 'MIDA', algorithm = 'Ridge', phenotypes = True, KHSIC = True
# seed = 123, connectivity = 'TPE', leave_one_out = False, filename = 'tangent'
# ensemble = False, validation_ext = '10CV'
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import time
import sklearn
from nilearn import connectome
import scipy.io as sio
from numpy.linalg import multi_dot
from sklearn.linear_model import LogisticRegression
import sklearn.svm as svm
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from prepdata import xxxxxx as Reader


class MIDA:
    def __init__(self, X, D, Y=None, mu=0.1, gamma_y=0.1, h=1035, labels=False):
        """
        :param X: All subjects feature matrix (n x m)
        :param D: domain feature matrix (n x num_domains)
        :param Y: label information matrix (n x 2)
        :param mu: covariance maximisation parameter
        :param gamma_y: dependence on label information paramter
        :param h: dimensionality of projected samples
        :param labels: unsupervised MIDA (False) or semi supervised MIDA (True)
        """
        super(MIDA, self).__init__()
        self.X = X
        self.D = D
        self.Y = Y
        self.mu = mu
        self.gamma_y = gamma_y
        self.h = h
        self.labels = labels

    # radial basis function (rbf) width parameter
    def width_rbf(self, X):
        n = X.shape[0]
        Xmed = X

        G = np.sum(Xmed * Xmed, 1).reshape(n, 1)
        Q = np.tile(G, (1, n))
        R = np.tile(G.T, (n, 1))

        dists = Q + R - 2 * np.dot(Xmed, Xmed.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n ** 2, 1)

        width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))

        return width_x

    # radial basis function (rbf) kernel matrix
    def rbf_dot(self, pattern1, pattern2, deg):
        size1 = pattern1.shape
        size2 = pattern2.shape

        G = np.sum(pattern1 * pattern1, 1).reshape(size1[0], 1)
        H = np.sum(pattern2 * pattern2, 1).reshape(size2[0], 1)

        Q = np.tile(G, (1, size2[0]))
        R = np.tile(H.T, (size1[0], 1))

        H = Q + R - 2 * np.dot(pattern1, pattern2.T)

        H = np.exp(-H / 2 / (deg ** 2))
        return H

    # Maximum Independene Domain Adaptation
    def cal_MIDA(self):
        """
        return:
            Z  : projected samples
        """

        # # Augment features with domain information
        X = np.concatenate([self.X, self.D], axis=1)

        # Augmented features rbf kernel
        width_x = self.width_rbf(X)
        K_x = self.rbf_dot(X, X, width_x)

        # site features linear kernel
        K_d = np.dot(self.D, self.D.T)

        # Centering matrix
        n = X.shape[0]
        H = np.identity(n) - (1 / n) * np.dot(np.ones((1, n)).T, np.ones((1, n)))

        if self.labels == False:
            # unsupervised MIDA

            mat = multi_dot([K_x, multi_dot([-1. * H, K_d, H]) + self.mu * H, K_x])
            eigs, eigv = np.linalg.eig(mat)
            ind = eigs.argsort()[-self.h:][::-1]
            W = eigv[:, ind]
        else:
            # semi supervised MIDA

            # label information linear kernel
            K_y = np.dot(self.Y, self.Y.T)

            mat = multi_dot([K_x, multi_dot([-1. * H, K_d, H]) + self.mu * H + self.gamma_y * multi_dot([H, K_y, H]), K_x])
            eigs, eigv = np.linalg.eig(mat)
            ind = eigs.argsort()[-self.h:][::-1]
            W = eigv[:, ind]

        # projected features
        Z = np.dot(W.T, K_x).T

        return Z

    def site_information_mat(self, data, num_subjects=1035, num_domains=20):

        Y = data[:, 1].reshape((num_subjects, 1))
        domain_features = np.zeros((num_subjects, num_domains))

        for i in range(num_subjects):
            domain_features[i, int(Y[i])] = 1

        return domain_features


class TrainABIDE:
    """
    10-fold cross validation for ABIDE
    leave one site out cross validation application performance for ABIDE
    leave one site out cross validation and ensemble models with different FC measures for ABIDE

    Args:
        subject_IDs (list): list of all subject IDs
        num_classes (int): number of classes for binary classification
        labels (dict): dict of labels of all subject
        data_folder (str): save data
        num_domains (int): number of sites available in the dataset
        model (str): Options: MIDA, raw. default: MIDA
        algorithm (str): Options: Ridge, LR (Logistic regression), SVM (Support vector machine). default: Ridge.
    """
    def __init__(self, subject_IDs, num_classes, labels, data_folder, num_domains, model='MIDA', algorithm='Ridge'):
        super(TrainABIDE, self).__init__()
        self.subject_IDs = subject_IDs
        self.num_classes = num_classes
        self.labels = labels
        self.data_folder = data_folder
        self.model = model
        self.num_domains = num_domains
        self.algorithm = algorithm

    # Transform test data using the transformer learned on the training data
    def process_test_data(self, timeseries, transformer, ids, atlas, connectivity, k, seed, validation_ext):
        conn_measure = connectome.ConnectivityMeasure(kind='correlation')
        test_data = conn_measure.fit_transform(timeseries)

        if connectivity == 'TE':
            connectivity = transformer.transform(timeseries)
        else:
            connectivity = transformer.transform(test_data)

        save_path = self.data_folder
        atlas_name = atlas
        kind = connectivity

        for i, subj_id in enumerate(ids):
            subject_file = os.path.join(save_path, subj_id,
                                        subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(
                                            k) + '_' + str(seed) + '_' + validation_ext + str(
                                            len(self.subject_IDs)) + '.mat')
            sio.savemat(subject_file, {'connectivity': connectivity[i]})

    # Process timeseries for tangent train/test split
    def process_timeseries(self, subject_IDs, train_ind, test_ind, atlas, connectivity, k, seed, validation_ext):
        timeseries = Reader.get_timeseries(subject_IDs, atlas, silence=True)
        train_timeseries = [timeseries[i] for i in train_ind]
        subject_IDs_train = [subject_IDs[i] for i in train_ind]
        test_timeseries = [timeseries[i] for i in test_ind]
        subject_IDs_test = [subject_IDs[i] for i in test_ind]

        print('computing tangent connectivity features..')
        transformer = Reader.subject_connectivity(train_timeseries, subject_IDs_train, atlas, connectivity, k, seed,
                                                  validation_ext, n_subjects=len(self.subject_IDs))
        test_data_save = self.process_test_data(test_timeseries, transformer, subject_IDs_test, atlas, connectivity,
                                                k, seed, validation_ext)

    # Grid search CV
    def grid_search(self, phenotypes, seed, train_ind, features, y, phenotype_ft=None, domain_ft=None):

        # MIDA parameter search
        mu_vals = [0.5, 0.75, 1.0]
        h_vals = [50, 150, 300]

        # Add phenotypes or not
        add_phenotypes = phenotypes

        # Algorithm choice
        algorithm = self.algorithm

        # Model choice
        model = self.model

        # best parameters and 5CV accuracy
        best_model = {}
        best_model['acc'] = 0

        # Grid search formulation
        if algorithm in ['LR', 'SVM']:
            C_vals = [1, 5, 10]
            if algorithm == 'LR':
                max_iter_vals = [100000]
                alg = LogisticRegression(random_state=seed, solver='lbfgs')
            else:
                max_iter_vals = [100000]
                alg = svm.SVC(random_state=seed, kernel='linear')
            parameters = {'C': C_vals, 'max_iter': max_iter_vals}
        else:
            alpha_vals = [0.25, 0.5, 0.75]
            parameters = {'alpha': alpha_vals}
            alg = RidgeClassifier(random_state=seed)

        if model in ['MIDA', 'SMIDA']:
            for mu in mu_vals:
                for h in h_vals:
                    x_data = features
                    MIDA_model = MIDA(x_data, domain_ft, mu=mu, h=h, labels=False)
                    x_data = MIDA_model.cal_MIDA()
                    if add_phenotypes == True:
                        x_data = np.concatenate([x_data, phenotype_ft], axis=1)
                    clf = GridSearchCV(alg, parameters, cv=5)
                    clf.fit(x_data[train_ind], y[train_ind].ravel())
                    if clf.best_score_ > best_model['acc']:
                        best_model['mu'] = mu
                        best_model['h'] = h
                        best_model = dict(best_model, **clf.best_params_)
                        best_model['acc'] = clf.best_score_

        else:
            x_data = features
            if add_phenotypes == True:
                x_data = np.concatenate([x_data, phenotype_ft], axis=1)
            clf = GridSearchCV(alg, parameters, cv=5)
            clf.fit(x_data[train_ind], y[train_ind].ravel())
            if clf.best_score_ > best_model['acc']:
                best_model = dict(best_model, **clf.best_params_)
                best_model['acc'] = clf.best_score_

        return best_model

    def train_10CV(self, phenotype_ft, phenotype_raw, seed=123, connectivity='TPE', atlas='cc200', phenotypes=True):
        """
        phenotype_ft: construct phenotype feature vectors.
        phenotype_raw: Source phenotype information and preprocess phenotypes.
        seed (int): Seed for random initialisation, set to 123 means no seed, default:123
        connectivity (str): Type of connectivity used for network construction.
                options: correlation, TE(tangent embedding), TPE(tangent pearson embedding), default: TPE.
        atlas (str): Atlas for network construction (node definition).
                options: ho, cc200, cc400, default: cc200.
        phenotypes (boolean): Add phenotype features. default: True.
        :return:
        """
        results_acc = []
        results_auc = []
        if seed == 123:
            skf = StratifiedKFold(n_splits=10, shuffle=True)
        else:
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        # Number of subjects for binary classification
        num_subjects = len(self.subject_IDs)

        # Initialise variables for class labels and acquisition sites
        y_data = np.zeros([num_subjects, self.num_classes])
        y = np.zeros([num_subjects, 1])

        # Get class labels for all subjects
        for i in range(num_subjects):
           y_data[i, int(self.labels[self.subject_IDs[i]]) - 1] = 1
           y[i] = int(self.labels[self.subject_IDs[i]])

        for sets, k in zip(list(reversed(list(skf.split(np.zeros(num_subjects), np.squeeze(y))))), list(range(10))):
            train_ind = sets[0]
            test_ind = sets[1]

            if connectivity in ['TPE', 'TE']:
                try:
                    features = Reader.get_networks(self.subject_IDs, iter_no=k, seed=seed, validation_ext='10CV',
                                                   kind=connectivity, n_subjects=num_subjects, atlas_name=atlas)
                except:
                    print("Tangent features not found. reloading timeseries data")
                    time.sleep(10)
                    self.process_timeseries(self.subject_IDs, train_ind, test_ind, atlas, connectivity, k, seed,
                                            '10CV')
                    features = Reader.get_networks(self.subject_IDs, iter_no=k, seed=seed, validation_ext='10CV',
                                                   kind=connectivity, n_subjects=num_subjects, atlas_name=atlas)

            if self.model == 'MIDA':
                domain_ft = MIDA.site_information_mat(phenotype_raw, num_subjects, self.num_domains)
                best_model = self.grid_search(phenotypes, seed, train_ind, features, y, phenotype_ft=phenotype_ft,
                                              domain_ft=domain_ft)
                print('best parameters from 5CV grid search: \n', best_model)
                MIDA_model = MIDA(features, domain_ft, mu=best_model['mu'], h=best_model['h'], labels=False)
                x_data = MIDA_model.cal_MIDA()
                best_model.pop('mu')
                best_model.pop('h')

            else:
                best_model = self.grid_search(phenotypes, seed, train_ind, features, y, phenotype_ft=phenotype_ft)
                print('best parameters from 5CV grid search: \n', best_model)
                x_data = features

            if phenotypes == True:
                x_data = np.concatenate([x_data, phenotype_ft], axis=1)

            # Remove accuracy key from best model dictionary
            best_model.pop('acc')

            # Set classifier
            if self.algorithm == 'LR':
                clf = LogisticRegression(random_state=seed, solver='lbfgs', **best_model)

            elif self.algorithm == 'SVM':
                clf = svm.SVC(random_state=seed, kernel='linear', **best_model)
            else:
                clf = RidgeClassifier(random_state=seed, **best_model)

            clf.fit(x_data[train_ind, :], y[train_ind].ravel())

            # Compute the accuracy
            lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())

            # Compute the AUC
            pred = clf.decision_function(x_data[test_ind, :])
            lin_auc = sklearn.metrics.roc_auc_score(y[test_ind], pred)

            # append accuracy and AUC to respective lists
            results_acc.append(lin_acc)
            results_auc.append(lin_auc)
            print("-" * 100)
            print("Fold number: %d" % k)
            print("Linear Accuracy: " + str(lin_acc))
            print("Linear AUC: " + str(lin_auc))
            print("-" * 100)

        return results_acc, results_auc, features, domain_ft


