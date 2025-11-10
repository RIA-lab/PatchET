from transformers import Trainer, PreTrainedModel
import os
import safetensors
import torch
from transformers.utils import is_peft_available
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
from transformers.utils import logging
from peft import PeftModel
from typing import Dict, Optional

TRAINING_ARGS_NAME = "training_args.bin"
logger = logging.get_logger(__name__)


class ModelTrainer(Trainer):
    def __init__(self, *args, test_dataset=None, test_metric_prefix='test', task=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_dataset = test_dataset
        self.test_metric_prefix = test_metric_prefix
        self.task = task
        self.save_pretrain_model = False

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Evaluate on validation set
        eval_metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # Evaluate on test set
        if self.test_dataset is not None and self.state.is_world_process_zero:
            test_output = self.predict(self.test_dataset, ignore_keys=ignore_keys)
            test_metrics = test_output.metrics
            test_metrics = {f"{self.test_metric_prefix}_{k}": v for k, v in test_metrics.items()}

            # Log test metrics
            self.log_metrics(self.test_metric_prefix, test_metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, test_metrics)

        return eval_metrics

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if not self.save_pretrain_model:
                state_dict = {k: v for k, v in state_dict.items() if k.split('.')[0] != 'pretrain_model'}

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

