import random
import timeit
import numpy as np
import pandas as pd
import torch
from dice_ml import diverse_counterfactuals as exp
from dice_ml.constants import ModelTypes
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
import pickle
from dice_ml.constants import ModelTypes
from dice_ml.utils.exception import SystemException
from dice_ml.utils.helpers import DataTransfomer

"""
    Module implementing custom DiCE Model objects for the various models used in experiments. By default, DiCE only
    works on Logistic Regression and custom prediction functions need to be passed in to handle unique models like
    TabNet, FTT and especially XGFTT. Base code obtained from the dice library.
"""


class XGFTModel:

    def __init__(self, tab, xg, func=None, kw_args=None):
        self.tab = tab
        self.xg = xg
        self.model_path = ''
        self.backend = 'sklearn'
        self.model_type = ModelTypes.Classifier
        self.model = xg
        # calls FunctionTransformer of scikit-learn internally
        # (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
        self.transformer = DataTransfomer(func, kw_args)

    def load_model(self):
        if self.model_path != '':
            with open(self.model_path, 'rb') as filehandle:
                self.model = pickle.load(filehandle)

    def get_output(self, input_instance, model_score=True):
        """returns prediction probabilities for a classifier and the predicted output for a regressor.

        :returns: an array of output scores for a classifier, and a singleton
        array of predicted value for a regressor.
        """
        input_instance = self.transformer.transform(input_instance)
        input_embeddings = self.tab.predict(input_instance)
        return self.xg.predict_proba(input_embeddings)

    def get_gradient(self):
        raise NotImplementedError

    def get_num_output_nodes(self, inp_size):
        temp_input = np.transpose(np.array([np.random.uniform(0, 1) for i in range(inp_size)]).reshape(-1, 1))
        return self.get_output(temp_input).shape[1]

    def get_num_output_nodes2(self, input_instance):
        if self.model_type == ModelTypes.Regressor:
            raise SystemException('Number of output nodes not supported for regression')
        return self.get_output(input_instance).shape[1]

class XGBoostModel:

    def __init__(self, model=None, model_path='', backend='', func=None, kw_args=None):
        """Init method

        :param model: trained ML Model.
        :param model_path: path to trained model.
        :param backend: ML framework. For frameworks other than TensorFlow or PyTorch,
                        or for implementations other than standard DiCE
                        (https://arxiv.org/pdf/1905.07697.pdf),
                        provide both the module and class names as module_name.class_name.
                        For instance, if there is a model interface class "SklearnModel"
                        in module "sklearn_model.py" inside the subpackage dice_ml.model_interfaces,
                        then backend parameter should be "sklearn_model.SklearnModel".
        :param func: function transformation required for ML model. If func is None, then func will be the identity function.
        :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the
                        dictionary of kw_args, by default.

        """
        self.model = model
        self.model_path = model_path
        self.backend = backend
        self.model_type = ModelTypes.Classifier
        # calls FunctionTransformer of scikit-learn internally
        # (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
        self.transformer = DataTransfomer(func, kw_args)

    def load_model(self):
        if self.model_path != '':
            with open(self.model_path, 'rb') as filehandle:
                self.model = pickle.load(filehandle)

    def get_output(self, input_instance, model_score=True):
        """returns prediction probabilities for a classifier and the predicted output for a regressor.

        :returns: an array of output scores for a classifier, and a singleton
        array of predicted value for a regressor.
        """
        input_instance = self.transformer.transform(input_instance).astype('int64')
        if model_score:
            if self.model_type == ModelTypes.Classifier:
                return self.model.predict_proba(input_instance)
            else:
                return self.model.predict(input_instance)
        else:
            return self.model.predict(input_instance)

    def get_gradient(self):
        raise NotImplementedError

    def get_num_output_nodes(self, inp_size):
        temp_input = np.transpose(np.array([np.random.uniform(0, 1) for i in range(inp_size)]).reshape(-1, 1))
        return self.get_output(temp_input).shape[1]

    def get_num_output_nodes2(self, input_instance):
        if self.model_type == ModelTypes.Regressor:
            raise SystemException('Number of output nodes not supported for regression')
        return self.get_output(input_instance).shape[1]

class TabNetModel:

    def __init__(self, model=None, model_path='', backend='', func=None, kw_args=None):
        """Init method

        :param model: trained ML Model.
        :param model_path: path to trained model.
        :param backend: ML framework. For frameworks other than TensorFlow or PyTorch,
                        or for implementations other than standard DiCE
                        (https://arxiv.org/pdf/1905.07697.pdf),
                        provide both the module and class names as module_name.class_name.
                        For instance, if there is a model interface class "SklearnModel"
                        in module "sklearn_model.py" inside the subpackage dice_ml.model_interfaces,
                        then backend parameter should be "sklearn_model.SklearnModel".
        :param func: function transformation required for ML model. If func is None, then func will be the identity function.
        :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the
                        dictionary of kw_args, by default.

        """
        self.model = model
        self.model_path = model_path
        self.backend = backend
        self.model_type = ModelTypes.Classifier
        # calls FunctionTransformer of scikit-learn internally
        # (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
        self.transformer = DataTransfomer(func, kw_args)

    def load_model(self):
        if self.model_path != '':
            with open(self.model_path, 'rb') as filehandle:
                self.model = pickle.load(filehandle)

    def get_output(self, input_instance, model_score=True):
        """returns prediction probabilities for a classifier and the predicted output for a regressor.

        :returns: an array of output scores for a classifier, and a singleton
        array of predicted value for a regressor.
        """

        input_instance = torch.tensor(input_instance.values)
        input_instance = self.transformer.transform(input_instance)
        if model_score:
            if self.model_type == ModelTypes.Classifier:
                return self.model.predict_proba(input_instance)
            else:
                return self.model.predict(input_instance)
        else:
            return self.model.predict(input_instance)

    def get_gradient(self):
        raise NotImplementedError

    def get_num_output_nodes(self, inp_size):
        temp_input = np.transpose(np.array([np.random.uniform(0, 1) for i in range(inp_size)]).reshape(-1, 1))
        return self.get_output(temp_input).shape[1]

    def get_num_output_nodes2(self, input_instance):
        if self.model_type == ModelTypes.Regressor:
            raise SystemException('Number of output nodes not supported for regression')
        return self.get_output(input_instance).shape[1]

class FTTModel:

    def __init__(self, model=None, model_path='', backend='', func=None, kw_args=None):
        """Init method

        :param model: trained ML Model.
        :param model_path: path to trained model.
        :param backend: ML framework. For frameworks other than TensorFlow or PyTorch,
                        or for implementations other than standard DiCE
                        (https://arxiv.org/pdf/1905.07697.pdf),
                        provide both the module and class names as module_name.class_name.
                        For instance, if there is a model interface class "SklearnModel"
                        in module "sklearn_model.py" inside the subpackage dice_ml.model_interfaces,
                        then backend parameter should be "sklearn_model.SklearnModel".
        :param func: function transformation required for ML model. If func is None, then func will be the identity function.
        :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the
                        dictionary of kw_args, by default.

        """
        self.model = model
        self.model_path = model_path
        self.backend = backend
        self.model_type = ModelTypes.Classifier
        # calls FunctionTransformer of scikit-learn internally
        # (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
        self.transformer = DataTransfomer(func, kw_args)

    def load_model(self):
        if self.model_path != '':
            with open(self.model_path, 'rb') as filehandle:
                self.model = pickle.load(filehandle)

    def get_output(self, input_instance, model_score=True):
        """returns prediction probabilities for a classifier and the predicted output for a regressor.

        :returns: an array of output scores for a classifier, and a singleton
        array of predicted value for a regressor.
        """
        input_instance = self.transformer.transform(input_instance)
        return self.model.predict(input_instance)[['0_probability', '1_probability']].values

    def get_gradient(self):
        raise NotImplementedError

    def get_num_output_nodes(self, inp_size):
        temp_input = np.transpose(np.array([np.random.uniform(0, 1) for i in range(inp_size)]).reshape(-1, 1))
        return self.get_output(temp_input).shape[1]

    def get_num_output_nodes2(self, input_instance):
        if self.model_type == ModelTypes.Regressor:
            raise SystemException('Number of output nodes not supported for regression')
        return self.get_output(input_instance).shape[1]
