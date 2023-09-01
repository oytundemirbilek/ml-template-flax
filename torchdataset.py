import os
from typing import Tuple, Optional, List
import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import Array
from torch import Tensor
from torch.utils.data import Dataset, default_collate
from sklearn.model_selection import KFold

jax.config.update("jax_platform_name", "gpu" if jax.device_count() > 0 else "cpu")

ROOT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, "datasets")

# We used 35813 (part of the Fibonacci Sequence) as the seed when we conducted experiments
np.random.seed(35813)


def mock_batch_collate_fn(batch_data: List[Tensor]) -> Tensor:
    """Definition of how the batched data will be treated."""
    return default_collate(batch_data)


class JaxDatasetBase(Dataset):
    """Base class for common functionalities of all datasets."""

    def __init__(
        self,
        path_to_data: str,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        in_memory: bool = False,
    ):
        """
        Dataset class to initialize data operations, cross validation and preprocessing.

        Parameters
        ----------
        path_to_data: string
            Path where the expected data is stored.
        mode: string
            Which split to return, can be 'train', 'validation', 'test', or 'inference'.
            Use 'inference' to load all data if you have a pretrained model and want to use
            it in-production.
        n_folds: integer
            Number of cross validation folds.
        current_fold: integer
            Defines which cross validation fold will be selected for training.
        in_memory: bool
            Whether to store all data in memory or not.
        """
        assert (
            current_fold < n_folds
        ), "selected fold index cannot be more than number of folds."
        self.mode = mode
        self.n_folds = n_folds
        self.in_memory = in_memory
        self.path_to_data = path_to_data
        self.current_fold = current_fold

        self.n_samples_total = self.get_number_of_samples()

        # Keep half of the data as 'unseen' to be used in inference.
        self.seen_data_indices, self.unseen_data_indices = self.get_fold_indices(
            self.n_samples_total, 2
        )

        if not mode == "inference" and in_memory:
            self.samples_labels = self.get_labels()

        if self.in_memory:
            self.loaded_samples = self.get_all_samples()

        if mode == "train" or mode == "validation":
            # Here split the 'seen' data to train and validation.
            if self.in_memory:
                self.seen_samples_labels = self.samples_labels[self.seen_data_indices]
                self.seen_samples_data = self.loaded_samples[self.seen_data_indices]

            self.n_samples_seen = len(self.seen_data_indices)
            self.tr_indices, self.val_indices = self.get_fold_indices(
                self.n_samples_seen,
                self.n_folds,
                self.current_fold,
            )

        if mode == "train":
            self.selected_indices = self.tr_indices
            if self.in_memory:
                self.samples_labels = self.seen_samples_labels[self.tr_indices]
                self.loaded_samples = self.seen_samples_data[self.tr_indices]
        elif mode == "validation":
            self.selected_indices = self.val_indices
            if self.in_memory:
                self.samples_labels = self.seen_samples_labels[self.val_indices]
                self.loaded_samples = self.seen_samples_data[self.val_indices]
        elif mode == "test":
            self.selected_indices = self.unseen_data_indices
            if self.in_memory:
                self.samples_labels = self.samples_labels[self.unseen_data_indices]
                self.loaded_samples = self.loaded_samples[self.unseen_data_indices]
        elif not mode == "inference":
            raise ValueError(
                "mode should be 'train', 'validation', 'test', or 'inference'"
            )
        if mode == "inference":
            self.n_samples_in_split = self.n_samples_total
        else:
            self.n_samples_in_split = len(self.selected_indices)

    def __getitem__(self, index: int) -> Tuple[Array, Optional[Array]]:
        if self.in_memory:
            if self.mode == "inference":
                label = None
            else:
                label = self.samples_labels[index]
            sample_data = self.loaded_samples[index]
        else:
            sample_data, label = self.get_sample_data(index)
        return self.preprocess(sample_data), label

    def __len__(self) -> int:
        return self.n_samples_in_split

    def get_number_of_samples(self) -> int:
        """
        Method to find how many samples are expected in ALL dataset.
        E.g., number of images in the target folder, number of rows in dataframe.
        """
        with open("./datasets/mock_dataset.csv") as fp:
            n_lines = len(fp.readlines())
        return n_lines - 1

    def get_labels(self) -> Array:
        """
        Method to read and store labels in a jax array

        Returns
        -------
        labels: jax Array
            An array stores the labels for each sample.
        """
        return jnp.asarray(pd.read_csv(self.path_to_data)["Label"].values)

    def get_sample_data(self, index: int) -> Tuple[Array, Array]:
        """
        If we cannot or do not want to store all the samples in memory, we need to
        read the data based on selected indices (train, validation or test).

        Parameters
        ----------
        index: integer
            In-split index of the expected sample.
            (e.g., 2 means the 3rd sample from validation split if mode is 'validation')

        Returns
        -------
        tensor: jax Array
            A jax array represents the data for the sample.
        """

        def skip_unselected(row_idx: int) -> bool:
            if row_idx == 0:
                return False
            return row_idx != (self.selected_indices[index] + 1)

        sample_data_row = pd.read_csv(self.path_to_data, skiprows=skip_unselected)
        sample_data = sample_data_row.drop(
            ["Sample ID", "Label"], axis="columns"
        ).values

        sample_label = sample_data_row["Label"].values
        return jnp.asarray(sample_data), jnp.asarray(sample_label, dtype=jnp.float32)

    def preprocess(self, data: Array) -> Array:
        return data

    def get_all_samples(self) -> Array:
        """
        Convert data from all samples to the jax Array objects to store in a list later.
        This function can be memory-consuming but time-saving, recommended to be used on small datasets.

        Returns
        -------
        all_data: jax Array
            A jax array represents all data.
        """
        return jnp.asarray(
            pd.read_csv(self.path_to_data).drop(["Sample ID", "Label"]).values
        )

    def get_fold_indices(
        self, all_data_size: int, n_folds: int, fold_id: int = 0
    ) -> Tuple[Array, Array]:
        """
        Create folds and get indices of train and validation datasets.

        Parameters
        --------
        all_data_size: int
            Size of all data.
        fold_id: int
            Which cross validation fold to get the indices for.

        Returns
        --------
        train_indices: jax Array
            Indices to get the training dataset.
        val_indices: jax Array
            Indices to get the validation dataset.
        """
        kf = KFold(n_splits=n_folds, shuffle=True)
        split_indices = kf.split(range(all_data_size))
        train_indices, val_indices = [
            (jnp.array(train), jnp.array(val)) for train, val in split_indices
        ][fold_id]
        # Split train and test
        return train_indices, val_indices

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} dataset ({self.mode}) with"
            f" n_samples={self.n_samples_in_split}, "
            f"current fold:{self.current_fold+1}/{self.n_folds}"
        )


class MockDataset(JaxDatasetBase):
    """
    Mock data
    """

    def __init__(
        self,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        in_memory: bool = False,
    ):
        mock_data_path = os.path.join(DATA_PATH, "mock_dataset.csv")
        super().__init__(
            mock_data_path,
            mode,
            n_folds,
            current_fold,
            in_memory,
        )
