"""
Dynamic.py

Contains the DynamicModel class, which is a wrapper around a machine learning model (e.g. Random Forest) that can be trained on a specific train split of the data. 
The DynamicModelManager class manages multiple DynamicModels, each trained on a different train split of the data.

"""
import copy
import time
import warnings

import pandas as pd
import pickle as pkl
import numpy as np

from typing import Optional, Protocol
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor


class AnyModel(Protocol):
    """
    Protocol for any machine learning model that implements fit and predict methods.
    """
    def fit(self, X, y) -> "AnyModel": 
        ...
    def predict(self, X) -> np.ndarray: 
        ...



@dataclass
class ModelConfig:
    name: str
    model: AnyModel
    queryset: str
    target: str
    features: list[str]


@dataclass(frozen=True)
class TrainSplit:
    """
    Dataclass for a training split of the data.

    Attributes:
        start_month (int): the starting month of the training split (inclusive)
        end_month (int): the ending month of the training split (inclusive)
        step (int): the number of months the model is predicting into the future
    """
    start_month: int
    end_month: int
    step: int


@dataclass(frozen=True)
class TestSplit:
    """
    Dataclass for a test split of the data.

    Attributes:
        start_month (int): the starting month of the training split (inclusive)
        end_month (int): the ending month of the training split (inclusive)
        step (int): the number of months the model is predicting into the future
    """
    start_month: int
    end_month: int
    step: int


class Prediction:
    """
    Class for storing predictions from a DynamicModel on a specific test split.

    Attributes:
        train_split (TrainSplit): the training split the model was trained on
        test_split (TestSplit): the test split the predictions are for
        predictions (pd.DataFrame): a dataframe containing the predictions, with columns 'target_month_id', 'country_id', and 'prediction'
        prediction_col (str): the name of the column in predictions that contains the predicted values
        prediction_target (str): the name of the target variable that the model is predicting (e.g. 'ucdp_ged_sb_best_sum')
    """

    def __init__(
            self,
            train_split: TrainSplit,
            test_split: TestSplit,
            predictions: pd.DataFrame,
            prediction_col: str = "prediction",
            prediction_target: str = "ucdp_ged_sb_best_sum",
            target_log_transformed: bool = False
        ):

        self.train_split = train_split
        self.test_split = test_split
        self.predictions = predictions
        self.prediction_col = prediction_col
        self.prediction_target = prediction_target
        self.target_log_transformed = target_log_transformed

        self.distance_ = self.calculate_distance()

    
    def calculate_distance(self) -> float:
        """
        Calculate the distance between the train and test splits in months.
        Distance is defined as the number of months between the closest edge of the train split to the middle of the test split. 
        If the train split overlaps with the test split, distance is not defined and an error is raised.
        
        Returns:
            float: the distance between the train and test splits in months
        """

        # compute the middle of the test split
        test_middle = (self.test_split.start_month + self.test_split.end_month) / 2
        
        # if the train split is after the test split, return the distance from the start of the train split to middle of the test split
        if self.train_split.start_month >= test_middle:
            return test_middle - self.train_split.start_month
        # if the train split is before the test split, return the distance from the end of the train split to middle of the test split
        elif self.train_split.end_month <= test_middle:
            return test_middle - self.train_split.end_month
        # if the train split overlaps with the test split, raise error
        else:
            raise ValueError("Train and test splits overlap, distance is not defined")


    def msle(
            self, 
            actuals: pd.DataFrame, 
        ) -> float:
        """
        Calculate Mean Squared Log Error.
        
        Arguments:
            actuals (pd.DataFrame): a dataframe containing the actual values, with columns 'month_id', 'country_id', 
            and the target variable (e.g. 'ucdp_ged_sb_best_sum'), corresponding to self.prediction_target.
        """

        # merge predictions with actuals on month_id and country_id
        merged = self.predictions.merge(
            actuals,
            left_on=["target_month_id", "country_id"],
            right_on=["month_id", "country_id"],
            how="inner"
        )

        # define y_true and y_pred
        y_true = merged[self.prediction_target].to_numpy(dtype=float)
        y_pred = merged[self.prediction_col].to_numpy(dtype=float)

        # if the target variable is log-transformed, we need to exponentiate the predictions and actuals before computing MSLE
        if self.target_log_transformed:
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)

        # check for negative values, as MSLE cannot be computed with negative values
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError("MSLE cannot be computed with negative values.")

        return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    

    def mse(
            self, 
            actuals: pd.DataFrame, 
        ) -> float:
        """
        Calculate Mean Squared Error.
        
        Arguments:
            actuals (pd.DataFrame): a dataframe containing the actual values, with columns 'month_id', 'country_id', 
            and the target variable (e.g. 'ucdp_ged_sb_best_sum'), corresponding to self.prediction_target.
        """

        # merge predictions with actuals on month_id and country_id
        merged = self.predictions.merge(
            actuals,
            left_on=["target_month_id", "country_id"],
            right_on=["month_id", "country_id"],
            how="inner"
        )

        # define y_true and y_pred
        y_true = merged[self.prediction_target].to_numpy(dtype=float)
        y_pred = merged[self.prediction_col].to_numpy(dtype=float)

        # if the target variable is log-transformed, we need to exponentiate the predictions and actuals before computing MSE
        if self.target_log_transformed:
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)

        # check for negative values, as MSE cannot be computed with negative values
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError("MSE cannot be computed with negative values.")

        return float(np.mean((y_true - y_pred) ** 2))


    def __repr__(self) -> str:
        return (
            f"Prediction({self.train_split}, "
            f"{self.test_split}, "
            f"Target: {self.prediction_target}, as column {self.prediction_col}, "
            f"Size={self.predictions.shape})"
        )

class DynamicModel:

    """
    Class for a Dynamic model that can be trained on a specific train split
    """

    def __init__(
            self, 
            train_split: TrainSplit,
            model: AnyModel,
        ):

        self.train_split = train_split
        self.model = model
        
        self.features = []
        self.target = ""

        self.train_shape_ = (0, 0)
        self._is_fitted = False


    def fit(
            self,
            data: pd.DataFrame, 
            features: list[str], 
            target: str,
         ) -> None:
        """
        Fit the model on the given data for the specified features and target variable, using the train split defined in self.train_split.
        The data must contain columns 'month_id', 'country_id', and the target variable defined in self.target, 
        as well as any features defined in self.features.

        Arguments:
            data (pd.DataFrame): a dataframe containing the data to fit the model on, which must contain columns 'month_id', 'country_id', 
                and the target variable defined in self.target, as well as any features defined in self.features.
            features (list[str]): a list of feature column names to be used for training the model.
            target (str): the name of the target variable column to be used for training the model.
        """

        data = data.copy()
 
        self.features = features
        self.target = target

        # lead the target variable by the step size
        data["target_month_id"] = data["month_id"] + self.train_split.step
        data["target"] = data.groupby("country_id")[target].shift(-self.train_split.step)

        # filter the data to only include rows where the target_month_id is within the train split
        data = data[data['target_month_id'].between(self.train_split.start_month, self.train_split.end_month)]

        # drop NAs
        data = data.dropna(subset=["target"])

        X = data[self.features]
        y = data["target"]

        self.train_shape_ = X.shape

        self.model.fit(X, y)

        self._is_fitted = True


    def predict(
            self, 
            data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate predictions for the given data using the fitted model.
        
        Arguments:
            data (pd.DataFrame): a dataframe containing the features to generate predictions for, which must
                contain columns 'month_id', 'country_id', and all features defined in self.features.
        Returns:
            np.ndarray: an array of predictions for the given data in the shape (n_samples,). 
        """
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")

        return self.model.predict(data)
    

    def __repr__(self):
        fitted = getattr(self, "_is_fitted", False)
        return (
            f"DynamicModel({self.train_split},"
            f"fitted={fitted})"
        )


class DynamicModelManager:

    """
    Class for managing multiple DynamicModels, each trained on a different train split of the data.

    Attributes:
        steps (list[int]): a list of step sizes (in months) that the models are predicting into the future, 
            which determines the train and test splits that are generated
        data (pd.DataFrame): the full dataset to be used for training and prediction, which
            must contain columns 'month_id', 'country_id', and the target variable defined in self.target, 
            as well as any features defined in self.features.
        features (list[str]): a list of feature column names to be used for training the models
        target (str): the name of the target variable column to be used for training the models
        train_window_size (int): the size of the training window in months
        test_window_size (int): the size of the test window in months
        slide_window_size (int): the size of the sliding window in months, which determines how much the train and test splits are shifted for each model
        full_split (tuple[int, int]): a tuple of (start_month, end_month) representing the full range of months in the dataset
        base_model (AnyModel): the machine learning model to be used for all DynamicModels, default is RandomForestRegressor
        default_random_state (int): the random state to be used for any random operations, default is 42
        target_log_transformed (bool): whether the target variable is log-transformed, default is False
        train_splits (list[TrainSplit]): a list of TrainSplit objects representing the train splits to be used for training the models, 
            generated based on the full split, train window size, and slide window size
        test_splits (list[TestSplit]): a list of TestSplit objects representing the test splits to be used for evaluation, 
            generated based on the full split, test window size, and slide window size
        models (dict[tuple[int, int], DynamicModel]): a dictionary of DynamicModel objects,
            where the keys are tuples of (start_month, step) for the train split, and the values are the fitted DynamicModel objects
        predictions (list[Prediction]): a list of Prediction objects representing the predictions made by the models on the test splits
    """

    def __init__(
            self, 
            steps: list[int], 
            data: pd.DataFrame,
            features: list[str], 
            target: str,
            full_split: tuple[int, int],
            model_name: str = "Dynamic Model",
            base_model: Optional[AnyModel] = None,
            train_window_size: int = 120, 
            test_window_size: int = 10,
            slide_window_size: int = 12,
            default_random_state: int = 42,
            target_log_transformed: bool = False
        ):
        
        self.data = data.copy()
        self.name = model_name
        self.features = features
        self.target = target
        self.steps = steps
        self.train_window_size = train_window_size 
        self.test_window_size = test_window_size 
        self.slide_window_size = slide_window_size 
        self.full_split = full_split

        self.target_log_transformed = target_log_transformed
        self._check_log_transformed()

        if base_model is None:
            self.base_model = RandomForestRegressor(random_state=default_random_state)
        else:
            self.base_model = base_model

        self.models: dict[tuple[int, int], DynamicModel] = {}
        self.predictions: list[Prediction] = []
        self._metrics: Optional[pd.DataFrame] = None
        self._is_fitted = False

        self.train_splits = self.get_train_splits()
        self.test_splits = self.get_test_splits()

    def fit(self) -> None:
        """
        Fit a DynamicModel for each train split and store them in self.models.
        The keys of self.models are tuples of (start_month, step) for the train split, and the values are the fitted DynamicModel objects.
        """

        start_time = time.time()
    
        for split in tqdm(self.train_splits, total = len(self.train_splits), desc="Fitting models"):
            
            # copy instance of model
            model_instance = copy.deepcopy(self.base_model)

            # init model
            model = DynamicModel(
                train_split=split, model=model_instance
            )

            # fit model
            model.fit(self.data, self.features, self.target)

            # save model
            self.models[(split.start_month, split.step)] = model

        # print elapsed time
        elapsed = time.time() - start_time
        print(f"Finished fitting all models in {elapsed:.2f} seconds")

        # set fitted flag to True
        self._is_fitted = True
        self._metrics = None # reset metrics, as they are no longer valid after fitting new models


    def save_artifact(
            self,
            path: str = "../data/model_artifacts"
        ) -> None:
        """
        Save the DynamicModelManager object as a pickle file in the data/model_artifacts directory.

        Arguments:
            path (str): the path to save the pickle file, default is "../data/model_artifacts"
        """

        with open(f"{path}/dynamic_model_manager.pkl", "wb") as f:
            pkl.dump(self, f)


    def predict(
            self, 
        ) -> None:
        """
        Generate predictions for all models on all test splits and store them in self.predictions as Prediction objects.
        Note, no predictions are made for test splits that overlap with the train split of the model.
        """

        data = self.data.copy()

        if not self._is_fitted:
            raise ValueError("Models must be fitted before predicting")

        # compute total number of iterations for progress bar
        total_iterations = int(len(self.models) * ((len(self.test_splits) / len(self.steps)) - (self.train_window_size // self.test_window_size)))
        
        with tqdm(total=total_iterations, desc="Predicting") as pbar:
            for model in self.models:

                model = self.models[model]

                for test_split in self.test_splits:

                    if self._has_overlap(model.train_split, test_split):
                        continue

                    if model.train_split.step != test_split.step:
                        continue
                    
                    test_data = data.copy()

                    # lead the target variable by the step size
                    test_data["target_month_id"] = test_data["month_id"] + test_split.step

                    # filter the data to only include rows where the month_id is within the test split
                    test_data = test_data[test_data['target_month_id'].between(test_split.start_month, test_split.end_month)]

                    # drop NA
                    test_data = test_data.dropna(subset=self.features)

                    X_test = test_data[self.features]

                    predictions = model.predict(X_test)

                    predictions_df = pd.DataFrame({
                        "target_month_id": test_data["target_month_id"],
                        "country_id": test_data["country_id"],
                        "prediction": predictions
                    })

                    prediction = Prediction(
                        train_split=model.train_split,
                        test_split=test_split,
                        predictions=predictions_df,
                        prediction_target=self.target,
                        target_log_transformed=self.target_log_transformed
                    )

                    self.predictions.append(prediction)

                    pbar.update(1)

    
    def get_train_splits(self) -> list[TrainSplit]:
        """
        Generate train splits based on the full split, train window size, and slide window size.
        
        Returns:
            list[TrainSplit]: a list of TrainSplit objects representing the train splits to be used for training the models.
        """

        train_splits = []

        # set the first month of the first training split
        month_min = self.full_split[0] + max(self.steps)
        # set the first month of the last training split
        month_max = self.full_split[1] - self.train_window_size
        
        for step in self.steps:

            start_month = month_min
            end_month = month_min + self.train_window_size - 1

            while start_month <= month_max:

                train_split = TrainSplit(start_month, end_month, step)

                start_month += self.slide_window_size
                end_month += self.slide_window_size

                train_splits.append(train_split)

        return train_splits
    

    def get_test_splits(
            self, 
        ) -> list[TestSplit]:
        """
        Generate test splits based on the full split, test window size, and slide window size.

        Returns:
            list[TestSplit]: a list of TestSplit objects representing the test splits to be used for evaluation.
        """

        test_splits = []

        # set the first month of the first training split
        month_min = self.full_split[0] + max(self.steps)
        # set the first month of the last training split
        month_max = self.full_split[1] - self.test_window_size

        for step in self.steps:

            start_month = month_min
            end_month = month_min + self.test_window_size - 1

            while start_month <= month_max:

                test_split = TestSplit(start_month, end_month, step)

                test_splits.append(test_split)

                start_month += self.slide_window_size
                end_month += self.slide_window_size

        return test_splits



    def get_metrics(self, future_only: bool = False) -> pd.DataFrame:
        """
        Compute a dataframe containing the MSE and MSLE for each prediction, as well as the distance between the train and test splits for each prediction.
        The dataframe has columns 'train_start', 'train_end', 'train_step', 'test_start', 'test_end', 'test_step', 'mse', 'msle', and 'distance'.

        Args:
            future_only (bool): whether to only include predictions where the test split is in the future relative to the train split, 
            i.e. where the distance between the train and test splits is positive. Default is False, 
            which includes all predictions regardless of the distance between the train and test splits.

        Returns:
            pd.DataFrame: a dataframe containing the MSE and MSLE for each prediction
        """

        if not self._is_fitted:
            raise ValueError("Models must be fitted and predictions must be generated before computing metrics")
        
        # if metrics have not been computed yet, compute them and store in self._metrics
        if self._metrics is None:
  
            # define actuals
            actuals = self.data[["country_id", "month_id", self.target]]

            # combine MSE and MSLE in dataframe for all predictions
            rows = []
            for prediction in self.predictions:
                row = {
                    "train_start": prediction.train_split.start_month,
                    "train_end": prediction.train_split.end_month,
                    "train_step": prediction.train_split.step,
                    "test_start": prediction.test_split.start_month,
                    "test_end": prediction.test_split.end_month,
                    "test_step": prediction.test_split.step,
                    "mse": prediction.mse(actuals),
                    "msle": prediction.msle(actuals),
                    "distance": prediction.distance_
                }
                rows.append(row)

            # combine into dataframe
            metrics_df = pd.DataFrame(rows)

            self._metrics = metrics_df

        # filter out predictions where the test split is in the past relative to the train split, if future_only is True
        if future_only:
            return self._metrics[self._metrics["distance"] >= 0] #type:ignore
        else:
            return self._metrics


    def get_disaggregated_metrics(self) -> pd.DataFrame:
        """
        Build a flat DataFrame containing all predictions, actuals, and their squared error.
        If the target of the manager is log transformed, calculates squared_log_error instead of squared_error.
        Each row represents one [country_id, month_id, train_split, test_split, step] combination

        Returns a DataFrame with columns:
            country_id, prediction, train_start, train_end, test_start, test_end, step, distance, month_id, ln_ged_sb_dep, squared_[log]_error
        """
        predictions = []
        actuals = self.data[["country_id", "month_id", self.target]]

        for pred in self.predictions:
            pred_dict = pred.predictions.copy()
            pred_dict['train_start'] = pred.train_split.start_month
            pred_dict['train_end'] = pred.train_split.end_month
            pred_dict['test_start'] = pred.test_split.start_month
            pred_dict['test_end'] = pred.test_split.end_month
            pred_dict['step'] = pred.train_split.step
            pred_dict['distance'] = pred.distance_

            predictions.append(pred_dict)

        full_predictions = pd.concat(predictions, ignore_index=True)

        merged = full_predictions.merge(
            actuals,
            left_on=["target_month_id", "country_id"],
            right_on=["month_id", "country_id"],
            how="inner"
        ).drop(columns=["target_month_id"])

        if self.target_log_transformed:
            error_type = "squared_log_error"
        else:
            error_type = "squared_error"

        merged[error_type] = (merged[self.target] - merged[pred.prediction_col]) ** 2

        return merged
    

    def _check_log_transformed(self) -> None:
        """
        Check if the target name hints at a log transformed variable, and if not already specified, raise a warning.
        """

        if "ln_" in self.target or "log_" in self.target:
            if not self.target_log_transformed:
                warnings.warn(
                    f"\nTarget variable '{self.target}' appears to be log-transformed "
                    f"but `target_log_transformed=False`.\n"
                    f"This may lead to incorrect metric calculations for MSE and MSLE, "
                    f"consider setting `target_log_transformed=True`.",
                    UserWarning,
                    stacklevel=2
                )
        
    

    @staticmethod
    def _has_overlap(train_split: TrainSplit, test_split: TestSplit) -> bool:
        """
        Check if the train split overlaps with the test split.
        Overlap is defined as any month that is included in both the train and test splits.
        
        Arguments:
            train_split (TrainSplit): the training split to check
            test_split (TestSplit): the test split to check
        Returns:
            bool: True if the train split overlaps with the test split, False otherwise
        """

        # if the train split is after the test split, there is no overlap
        if train_split.start_month >= test_split.end_month:
            return False
        # if the train split is before the test split, there is no overlap
        elif train_split.end_month <= test_split.start_month:
            return False
        # if the train split overlaps with the test split, return True
        else:
            return True