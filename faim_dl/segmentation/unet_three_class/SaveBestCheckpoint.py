import textwrap
from typing import Callable, Optional, Union

from composer import Callback, Logger
from composer.core.time import Time
from composer.core import Event, State
from composer.utils import checkpoint, dist
from composer.callbacks.checkpoint_saver import checkpoint_periodically
import os


class SaveBestCheckpoint(Callback):
    def __init__(
            self,
            save_folder: str = "checkpoints",
            overwrite: bool = False,
            save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = "1ep",
            weights_only: bool = False,
    ):

        if not callable(save_interval):
            save_interval = checkpoint_periodically(save_interval)

        self.checkpoint_folder = save_folder
        self.overwrite = overwrite

        self.save_interval = save_interval
        self.saved_checkpoints = {}
        self.weights_only = weights_only

        self.value = 0
        self.best_value = self.value

    def init(self, state: State, logger: Logger) -> None:
        # Each rank will attempt to create the checkpoint folder.
        # If the folder is not parameterized by rank, then exist_ok must be True, as the folder will be the same on all ranks.
        os.makedirs(self.checkpoint_folder, mode=0o775, exist_ok=True)
        if not self.overwrite:
            if any(x.startswith(".") for x in os.listdir(self.checkpoint_folder)):
                raise RuntimeError(
                    textwrap.dedent(f"""\
                    Checkpoint folder {self.checkpoint_folder} is not empty. When using {type(self).__name__}(overwrite=True, ...),
                    the checkpoint folder must not contain any existing checkpoints."""))
        # Ensure no rank proceeds (and potentially attempts to write to the folder), until all ranks have validated that the folder is empty.
        dist.barrier()

    def fit_start(self, state: State, logger: Logger) -> None:
        if state.is_model_deepspeed:
            if self.weights_only:
                NotImplementedError(
                    textwrap.dedent(f"""\
                    Saving checkpoints with `weights_only=True` is not currently supported when using DeepSpeed.
                    See https://github.com/mosaicml/composer/issues/685."""))

    def eval_batch_end(self, state: State, logger: Logger):
        self.value = state.evaluators[0].metrics.compute()['Dice']

    def batch_checkpoint(self, state: State, logger: Logger):
        del logger

        if self.save_interval(state, Event.BATCH_CHECKPOINT):
            self._save_checkpoint(state)

    def epoch_checkpoint(self, state: State, logger: Logger):
        del logger

        if self.save_interval(state, Event.EPOCH_CHECKPOINT):
            self._save_checkpoint(state)

    def _save_checkpoint(self, state: State):
        latest_checkpoint_file_path_format = os.path.join(self.checkpoint_folder, "latest_model")
        checkpoint.save_checkpoint(state, latest_checkpoint_file_path_format, weights_only=self.weights_only)
        timestamp = state.timer.get_timestamp()
        self.saved_checkpoints["latest"] = latest_checkpoint_file_path_format

        if self.value > self.best_value:
            self.best_value = self.value
            checkpoint_filepath_format = os.path.join(self.checkpoint_folder, "best_model")
            checkpoint.save_checkpoint(state, checkpoint_filepath_format, weights_only=self.weights_only)
            timestamp = state.timer.get_timestamp()
            self.saved_checkpoints["best"] = checkpoint_filepath_format
