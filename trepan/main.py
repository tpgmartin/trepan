import numpy as np
from scipy import stats

class Oracle(object):

    def __init__(self, network, num_classes, X_train):
        self.network = network
        self.num_classes = num_classes
        self.dimension = X_train.shape[1]
        self.feature_distributions = self.generate_feature_distributions(X_train)
    
    def generate_feature_distributions(self, X_train):

        return [stats.gaussian_kde(X_train[:,i].reshape(X_train.shape[0], bw_method="silverman")) for i in range(0, self.dimension)]
    
    def get_oracle_label(self, example):
        # Assumes certain API for model - for time being will be keras

        return np.argmax(self.network.predict(np.array([example])).reshape(self.num_classes))

    
    def generate_constrained_examples_with_labels(self, constraints, num_examples):

        pass

        # TODO: return generated examples and oracle label
        # examples is a 2D array, labels are 1D

        # oracle_labels = [self.get_oracle_label(example) for ]
        self.generate_constrained_example(constraints)

    def generate_constrained_example(self, constraints):

        pass

        # This references self.feature_distributions
