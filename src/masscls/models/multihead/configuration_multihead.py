from typing import List, Optional, Dict

from transformers import PretrainedConfig


class MultiHeadCNNConfig(PretrainedConfig):
    model_type = "multiheadcnn"

    def __init__(
        self,
        class_weights: Dict[str, List[float]] = {},
        backbone: str = "convnext_base.fb_in22k_ft_in1k",
        head_dims: Optional[List[List[int]]] = None,
        head_names: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super(MultiHeadCNNConfig, self).__init__(**kwargs)
        self.backbone = backbone
        self.attention = "cbam"

        if head_dims is None:
            self.head_dims = [[1024, 512, 256, 128]] * 4
        else:
            self.head_dims = head_dims

        if head_names is None:
            self.head_names = {f"C{k}": 2 for k in range(len(self.head_dims))}
        else:
            self.head_names = head_names

        self.num_classes = [self.head_names[k] for k in self.head_names]

        self.class_weights = class_weights
        self.gamma = kwargs.get("gamma", {k: 2.0 for k in self.head_names})
        self.smoothing = kwargs.get("smoothing", {k: 0.1 for k in self.head_names})
        self.task_weights = kwargs.get(
            "task_weights", {k: 1.0 for k in self.head_names}
        )


__all__ = ["MultiHeadCNNConfig"]
