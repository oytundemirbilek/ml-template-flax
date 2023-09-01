# Option to make CUDA runtime reproducible (If you are using gpu)
# https://github.com/google/jax/issues/4823
XLA_FLAGS = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
import json
import os
from typing import Any, Dict, Optional, List

import numpy as np
from flax.training import train_state, checkpoints, early_stopping
import jax
import jax.numpy as jnp
import optax
from torch.utils.data import DataLoader

from torchdataset import MockDataset, mock_batch_collate_fn
from model import MockModel
from evaluation import MockLoss

# We used 35813 (part of the Fibonacci Sequence) as the seed when we conducted experiments
np.random.seed(35813)
JAX_RANDOM_KEY = jax.random.PRNGKey(35813)


jax.config.update("jax_platform_name", "gpu" if jax.device_count() > 0 else "cpu")

FILE_PATH = os.path.dirname(__file__)

DATASETS = {
    "mock_dataset": MockDataset,
    "another_mock_dataset": MockDataset,
}


class BaseTrainer:
    """Wrapper around training function to save all the training parameters"""

    def __init__(
        self,
        # Data related:
        dataset: str,
        timepoint: Optional[str],
        # Training related:
        n_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.001,
        batch_size: int = 1,
        validation_period: int = 5,
        patience: int = 0,
        # Model related:
        n_folds: int = 5,
        in_features: int = 3,
        out_features: int = 3,
        layer_sizes: List[int] = [8, 16, 32],
        loss_weight: float = 1.0,
        loss_name: str = "mock_loss",
        model_name: str = "default_model_name",
    ) -> None:
        self.dataset = dataset
        self.timepoint = timepoint
        self.n_epochs = n_epochs
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_period = validation_period
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        self.in_features = in_features
        self.out_features = out_features
        self.layer_sizes = layer_sizes
        self.model_name = model_name
        self.model_save_path = os.path.join(FILE_PATH, "models", model_name)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.model_params_save_path = os.path.join(
            FILE_PATH, "models", model_name + "_params.json"
        )
        with open(self.model_params_save_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

        if loss_name == "mock_loss":
            self.loss_fn = MockLoss()
        else:
            raise NotImplementedError("Specified loss function is not defined.")

        self.val_loss_per_epoch: List[float] = []

    def __repr__(self) -> str:
        return str(self.__dict__)

    def train_state_init(self) -> train_state.TrainState:
        expected_shape = [
            self.batch_size,
            self.in_features,
        ]
        optimizer = optax.adamw(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )
        rng, init_rng = jax.random.split(JAX_RANDOM_KEY)

        model = MockModel(
            in_features=self.in_features,
            out_features=self.out_features,
            batch_size=self.batch_size,
            layer_sizes=self.layer_sizes,
        )
        params = model.init(init_rng, jnp.ones(expected_shape))["params"]
        return train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        )

    @jax.jit
    def validate(
        self, model_params: Dict[str, Any], val_dataloader: DataLoader
    ) -> float:
        val_losses = []
        for input_data, target_label in val_dataloader:
            pred_data = MockModel(
                in_features=self.in_features,
                out_features=self.out_features,
                batch_size=self.batch_size,
                layer_sizes=self.layer_sizes,
            ).apply({"params": model_params}, input_data)

            val_loss = self.loss_fn.compute(pred_data, target_label)
            val_losses.append(val_loss)

        return jnp.stack(val_losses).mean().item()

    @jax.jit
    def train(self, current_fold: int = 0) -> train_state.TrainState:
        tr_dataset = DATASETS[self.dataset](
            mode="train",
            n_folds=self.n_folds,
            current_fold=current_fold,
        )
        val_dataset = DATASETS[self.dataset](
            mode="validation",
            n_folds=self.n_folds,
            current_fold=current_fold,
        )

        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size=self.batch_size,
            collate_fn=mock_batch_collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=mock_batch_collate_fn,
        )

        state = self.train_state_init()

        early_stop = early_stopping.EarlyStopping(
            min_delta=1e-3, patience=self.patience
        )
        best_loss = 99999999999999999999999.0
        for epoch in range(self.n_epochs):
            tr_losses = []
            for input_data, target_data in tr_dataloader:
                pred_data = MockModel(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    batch_size=self.batch_size,
                    layer_sizes=self.layer_sizes,
                ).apply({"params": state.params}, input_data)

                grad_fn = jax.value_and_grad(self.loss_fn.compute, has_aux=True)
                (tr_loss,), grads = grad_fn(pred_data, target_data)
                state = state.apply_gradients(grads=grads)
                tr_losses.append(tr_loss)

            avg_tr_loss = jnp.stack(tr_losses).mean().item()
            if (epoch + 1) % self.validation_period == 0:
                val_loss = self.validate(state.params, val_dataloader)
                print(
                    f"Epoch: {epoch+1}/{self.n_epochs} | Tr.Loss: {avg_tr_loss} | Val.Loss: {val_loss}"
                )
                self.val_loss_per_epoch.append(val_loss)
                _, early_stop = early_stop.update(val_loss)
                if early_stop.should_stop:
                    break

                if val_loss < best_loss:
                    checkpoints.save_checkpoint(
                        ckpt_dir=os.path.join(
                            self.model_save_path, f"fold{current_fold}.pth"
                        ),
                        target=state,
                        step=epoch,
                    )
                    best_loss = val_loss

        return state

    def select_model(self) -> None:
        """A post processing method to combine trained cross validation models to be used later for inference."""
        return


if __name__ == "__main__":
    trainer = BaseTrainer(
        dataset="mock_dataset",
        timepoint=None,
        n_epochs=100,
        learning_rate=0.01,
    )
    trainer.train()
