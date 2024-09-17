import copy
import warnings
from functools import partial
from typing import List, Optional, Any
import torch
from omegaconf.dictconfig import DictConfig
from pandas import DataFrame
from pytorch_tabular import TabularModel
from torch import Tensor

from pytorch_tabular.utils import (
    get_logger
)

try:
    import captum.attr

    CAPTUM_INSTALLED = True
except ImportError:
    CAPTUM_INSTALLED = False

logger = get_logger(__name__)

"""
    The following class is a custom adaptation of the FT-Transformer to implement the XGFT-Transformer architecture
    including beheading and custom predict function. It inherits most components from the normal tabular model aside from
    the functions altered below. Base code obtained from the original pytorch-tabular library.
"""


class CustomTabularModel(TabularModel):
    def __init__(
        self,
        config: Optional[DictConfig] = None,
        **kwargs
    ) -> None:
        """The core model which orchestrates everything from initializing the datamodule, the model, trainer, etc.

        Args:
            config (Optional[Union[DictConfig, str]], optional): Single OmegaConf DictConfig object or
                the path to the yaml file holding all the config parameters. Defaults to None.

            data_config (Optional[Union[DataConfig, str]], optional):
                DataConfig object or path to the yaml file. Defaults to None.

            model_config (Optional[Union[ModelConfig, str]], optional):
                A subclass of ModelConfig or path to the yaml file.
                Determines which model to run from the type of config. Defaults to None.

            optimizer_config (Optional[Union[OptimizerConfig, str]], optional):
                OptimizerConfig object or path to the yaml file. Defaults to None.

            trainer_config (Optional[Union[TrainerConfig, str]], optional):
                TrainerConfig object or path to the yaml file. Defaults to None.

            experiment_config (Optional[Union[ExperimentConfig, str]], optional):
                ExperimentConfig object or path to the yaml file.
                If Provided configures the experiment tracking. Defaults to None.

            model_callable (Optional[Callable], optional):
                If provided, will override the model callable that will be loaded from the config.
                Typically used when providing Custom Models

            model_state_dict_path (Optional[Union[str, Path]], optional):
                If provided, will load the state dict after initializing the model from config.

            verbose (bool): turns off and on the logging. Defaults to True.

            suppress_lightning_logger (bool): If True, will suppress the default logging from PyTorch Lightning.
                Defaults to False.
        """
        super().__init__(config, **kwargs)
        self.beheaded = False

    # Removes the final linear head layers of the model
    def behead_model(self):
        sd = copy.deepcopy(self.model.state_dict())
        del sd['_head.layers.2.weight']
        del sd['_head.layers.2.bias']
        self.model.head.layers = self.model.head.layers[:1]
        self.model.load_state_dict(sd)
        self.beheaded = True

    def _predict(
        self,
        test: DataFrame,
        quantiles: Optional[List] = [0.25, 0.5, 0.75],
        n_samples: Optional[int] = 100,
        ret_logits=False,
        include_input_features: bool = False,
        device: Optional[torch.device] = None,
        progress_bar: Optional[str] = None
    ) -> Tensor | DataFrame | Any:
        """Uses the trained model to predict on new data and return as a dataframe.

        Args:
            test (DataFrame): The new dataframe with the features defined during training
            quantiles (Optional[List]): For probabilistic models like Mixture Density Networks, this specifies
                the different quantiles to be extracted apart from the `central_tendency` and added to the dataframe.
                For other models it is ignored. Defaults to [0.25, 0.5, 0.75]
            n_samples (Optional[int]): Number of samples to draw from the posterior to estimate the quantiles.
                Ignored for non-probabilistic models. Defaults to 100
            ret_logits (bool): Flag to return raw model outputs/logits except the backbone features along
                with the dataframe. Defaults to False
            include_input_features (bool): DEPRECATED: Flag to include the input features in the returned dataframe.
                Defaults to True
            progress_bar: choose progress bar for tracking the progress. "rich" or "tqdm" will set the respective
                progress bars. If None, no progress bar will be shown.

        Returns:
            DataFrame: Returns a dataframe with predictions and features (if `include_input_features=True`).
                If classification, it returns probabilities and final prediction
        """
        assert all(q <= 1 and q >= 0 for q in quantiles), "Quantiles should be a decimal between 0 and 1"
        model = self.model  # default
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if self.model.device != device:
                model = self.model.to(device)
        model.eval()
        inference_dataloader = self.datamodule.prepare_inference_dataloader(test)
        is_probabilistic = hasattr(model.hparams, "_probabilistic") and model.hparams._probabilistic

        if progress_bar == "rich":
            from rich.progress import track

            progress_bar = partial(track, description="Generating Predictions...")
        elif progress_bar == "tqdm":
            from tqdm.auto import tqdm

            progress_bar = partial(tqdm, description="Generating Predictions...")
        else:
            progress_bar = lambda it: it  # noqa E731
        point_predictions, quantile_predictions, logits_predictions = self._generate_predictions(
            model,
            inference_dataloader,
            quantiles,
            n_samples,
            ret_logits,
            progress_bar,
            is_probabilistic,
        )

        # If the model is beheaded, return embeddings early
        if self.beheaded:
            return point_predictions

        pred_df = self._format_predicitons(
            test,
            point_predictions,
            quantile_predictions,
            logits_predictions,
            quantiles,
            ret_logits,
            include_input_features,
            is_probabilistic,
        )
        return pred_df

    def predict(
        self,
        test: DataFrame,
        quantiles: Optional[List] = [0.25, 0.5, 0.75],
        n_samples: Optional[int] = 100,
        ret_logits=False,
        include_input_features: bool = False,
        device: Optional[torch.device] = None,
        progress_bar: Optional[str] = None,
        test_time_augmentation: Optional[bool] = False,
        num_tta: Optional[float] = 5,
        alpha_tta: Optional[float] = 0.1,
        aggregate_tta: Optional[str] = "mean",
        tta_seed: Optional[int] = 42
    ) -> DataFrame:
        """Uses the trained model to predict on new data and return as a dataframe.

        Args:
            test (DataFrame): The new dataframe with the features defined during training

            quantiles (Optional[List]): For probabilistic models like Mixture Density Networks, this specifies
                the different quantiles to be extracted apart from the `central_tendency` and added to the dataframe.
                For other models it is ignored. Defaults to [0.25, 0.5, 0.75]

            n_samples (Optional[int]): Number of samples to draw from the posterior to estimate the quantiles.
                Ignored for non-probabilistic models. Defaults to 100

            ret_logits (bool): Flag to return raw model outputs/logits except the backbone features along
                with the dataframe. Defaults to False

            include_input_features (bool): DEPRECATED: Flag to include the input features in the returned dataframe.
                Defaults to True

            progress_bar: choose progress bar for tracking the progress. "rich" or "tqdm" will set the respective
                progress bars. If None, no progress bar will be shown.

            test_time_augmentation (bool): If True, will use test time augmentation to generate predictions.
                The approach is very similar to what is described [here](https://kozodoi.me/blog/20210908/tta-tabular)
                But, we add noise to the embedded inputs to handle categorical features as well.\
                \\(x_{aug} = x_{orig} + \alpha * \\epsilon\\) where \\(\\epsilon \\sim \\mathcal{N}(0, 1)\\)
                Defaults to False
            num_tta (float): The number of augumentations to run TTA for. Defaults to 0.0

            alpha_tta (float): The standard deviation of the gaussian noise to be added to the input features

            aggregate_tta (Union[str, Callable], optional): The function to be used to aggregate the
                predictions from each augumentation. If str, should be one of "mean", "median", "min", or "max"
                for regression. For classification, the previous options are applied to the confidence
                scores (soft voting) and then converted to final prediction. An additional option
                "hard_voting" is available for classification.
                If callable, should be a function that takes in a list of 3D arrays (num_samples, num_cv, num_targets)
                and returns a 2D array of final probabilities (num_samples, num_targets). Defaults to "mean".'

            tta_seed (int): The random seed to be used for the noise added in TTA. Defaults to 42.

        Returns:
            DataFrame: Returns a dataframe with predictions and features (if `include_input_features=True`).
                If classification, it returns probabilities and final prediction
        """
        warnings.warn(
            "`include_input_features` will be deprecated in the next release."
            " Please add index columns to the test dataframe if you want to"
            " retain some features like the key or id",
            DeprecationWarning,
        )
        if test_time_augmentation:
            assert num_tta > 0, "num_tta should be greater than 0"
            assert alpha_tta > 0, "alpha_tta should be greater than 0"
            assert include_input_features is False, "include_input_features cannot be True for TTA."
            if not callable(aggregate_tta):
                assert aggregate_tta in [
                    "mean",
                    "median",
                    "min",
                    "max",
                    "hard_voting",
                ], (
                    "aggregate should be one of 'mean', 'median', 'min', 'max', or" " 'hard_voting'"
                )
            if self.config.task == "regression":
                assert aggregate_tta != "hard_voting", "hard_voting is only available for classification"

            torch.manual_seed(tta_seed)

            def add_noise(module, input, output):
                return output + alpha_tta * torch.randn_like(output, memory_format=torch.contiguous_format)

            # Register the hook to the embedding_layer
            handle = self.model.embedding_layer.register_forward_hook(add_noise)
            pred_prob_l = []
            for _ in range(num_tta):
                pred_df = self._predict(
                    test,
                    quantiles,
                    n_samples,
                    ret_logits,
                    include_input_features=False,
                    device=device,
                    progress_bar=progress_bar or "None"
                )
                pred_idx = pred_df.index
                if self.config.task == "classification":
                    pred_prob_l.append(pred_df.values[:, : -len(self.config.target)])
                elif self.config.task == "regression":
                    pred_prob_l.append(pred_df.values)
            pred_df = self._combine_predictions(pred_prob_l, pred_idx, aggregate_tta, None)
            # Remove the hook
            handle.remove()
        else:
            pred_df = self._predict(
                test,
                quantiles,
                n_samples,
                ret_logits,
                include_input_features,
                device,
                progress_bar,
            )
        return pred_df
