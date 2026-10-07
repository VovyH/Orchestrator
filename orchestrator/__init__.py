"""轻量训练调度。不写 CUDA/PPU kernel，只调度 packing、精度、编译和分块 loss。"""

__all__ = ["train", "load_config", "fit", "FitResult", "train_model", "TrainConfig", "TrainResult"]


def __getattr__(name: str):
    if name == "train":
        from .trainer import train

        return train
    if name in {"train_model", "TrainResult"}:
        from . import trainer

        globals()["train_model"] = trainer.train_model
        globals()["TrainResult"] = trainer.TrainResult
        return getattr(trainer, name)
    if name == "load_config":
        from .config import load_config

        return load_config
    if name == "TrainConfig":
        from .config import TrainConfig

        return TrainConfig
    if name in {"fit", "FitResult"}:
        from . import api

        globals()["api"] = api
        globals()["fit"] = api.fit
        globals()["FitResult"] = api.FitResult
        return getattr(api, name)
    raise AttributeError(name)
