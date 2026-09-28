#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


class TrainingLogger:
    """Unified training logger supporting TensorBoard and Weights & Biases."""

    SUPPORTED_BACKENDS = ("none", "tensorboard", "wandb")

    def __init__(self, args):
        self.backend = str(getattr(args, "logger_backend", "none") or "none").lower()
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported logger_backend={self.backend!r}. "
                f"Choose from {self.SUPPORTED_BACKENDS}."
            )

        self.log_interval = int(getattr(args, "logger_log_interval", 10))
        self.log_images = bool(getattr(args, "logger_log_images", True))
        self.model_path = args.model_path
        self._writer = None
        self._wandb = None

        if self.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(args.model_path, "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir)
            print(f"[Logger] TensorBoard logs: {log_dir}")
        elif self.backend == "wandb":
            import wandb

            run_name = getattr(args, "wandb_run_name", None)
            if not run_name:
                run_name = os.path.basename(os.path.normpath(args.model_path))

            config = None
            try:
                from omegaconf import OmegaConf

                config = OmegaConf.to_container(args, resolve=True)
            except Exception:
                config = dict(args)

            self._wandb = wandb.init(
                project=getattr(args, "wandb_project", "gs-lidar"),
                entity=getattr(args, "wandb_entity", None),
                name=run_name,
                config=config,
                dir=args.model_path,
            )
            print(f"[Logger] W&B run: {self._wandb.url}")

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def log_scalars(self, scalars: Dict[str, Any], step: int, prefix: str = "train") -> None:
        if not self.enabled:
            return

        payload = {}
        for key, value in scalars.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item() if value.numel() == 1 else None
            if value is None or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float, np.floating, np.integer)):
                continue
            tag = f"{prefix}/{key}" if prefix else key
            payload[tag] = float(value)

        if not payload:
            return

        if self.backend == "tensorboard":
            for tag, value in payload.items():
                self._writer.add_scalar(tag, value, step)
        elif self.backend == "wandb":
            self._wandb.log(payload, step=step)

    def log_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.ndarray],
        step: int,
    ) -> None:
        if not self.enabled or not self.log_images:
            return

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                image = image[:3]
            image = image.clamp(0.0, 1.0).numpy()

        if self.backend == "tensorboard":
            self._writer.add_image(tag, image, step, dataformats="CHW")
        elif self.backend == "wandb":
            self._wandb.log({tag: self._wandb.Image(image)}, step=step)

    def log_train_step(self, iteration: int, log_dict: Dict[str, Any]) -> None:
        if iteration % self.log_interval != 0:
            return
        self.log_scalars(log_dict, iteration, prefix="train")

    def log_eval_metrics(
        self,
        iteration: int,
        split: str,
        metrics: Dict[str, Dict[str, Any]],
    ) -> None:
        flat = {}
        for category, values in metrics.items():
            for metric_name, value in values.items():
                flat[f"{category}/{metric_name}"] = value
        self.log_scalars(flat, iteration, prefix=f"eval/{split}")

    def close(self) -> None:
        if self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()
            self._writer = None
        elif self.backend == "wandb" and self._wandb is not None:
            import wandb

            wandb.finish()
            self._wandb = None
