"""轻量训练调度。不写 CUDA/PPU kernel，只调度 packing、精度、编译和分块 loss。"""

__all__ = ["train", "load_config"]


def __getattr__(name: str):
    if name == "train":
        from .trainer import train

        return train
    if name == "load_config":
        from .config import load_config

        return load_config
    raise AttributeError(name)
