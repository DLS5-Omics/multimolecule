from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Optional

from transformers.configuration_utils import PretrainedConfig as _PretrainedConfig


class PretrainedConfig(_PretrainedConfig):
    head: HeadConfig

    def __init__(
        self, pad_token_id=0, bos_token_id=1, eos_token_id=2, unk_token_id=3, mask_token_id=4, null_token_id=5, **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            mask_token_id=mask_token_id,
            null_token_id=null_token_id,
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        for k, v in output.items():
            if hasattr(v, "to_dict"):
                output[k] = v.to_dict()
            if is_dataclass(v):
                output[k] = asdict(v)
        return output


@dataclass
class HeadConfig:
    r"""
    This is the configuration class to store the configuration of a prediction head. It is used to instantiate a
    prediction head according to the specified arguments, defining the head architecture.

    Configuration objects inherit from [`HeadConfig`] and can be used to control the model outputs. Read the
    documentation from [`HeadConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden states.
        transform (`str`, *optional*, defaults to None):
            The transform operation applied to hidden states.
        transform_act (`str`, *optional*, defaults to "tanh"):
            The activation function of transform applied to hidden states.
        bias (`bool`, *optional*, defaults to True):
            Whether to apply bias to the final prediction layer.
        act (`str`, *optional*, defaults to None):
            The activation function of the final prediction output.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        num_labels (`int`, *optional*, defaults to 2):
            Number of labels to use in the last layer added to the model, typically for a classification task.
        problem_type (`str`, *optional*):
            Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`,
            `"single_label_classification"` or `"multi_label_classification"`.
    """

    hidden_size: Optional[int] = None
    dropout: float = 0.0
    transform: Optional[str] = None
    transform_act: Optional[str] = "tanh"
    bias: bool = True
    act: Optional[str] = None
    layer_norm_eps: float = 1e-12
    num_labels: int = 2
    problem_type: Optional[str] = None


@dataclass
class MaskedLMHeadConfig:
    r"""
    This is the configuration class to store the configuration of a prediction head. It is used to instantiate a
    prediction head according to the specified arguments, defining the head architecture.

    Configuration objects inherit from [`HeadConfig`] and can be used to control the model outputs. Read the
    documentation from [`HeadConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden states.
        transform (`str`, *optional*, defaults to "nonlinear"):
            The transform operation applied to hidden states.
        transform_act (`str`, *optional*, defaults to "gelu"):
            The activation function of transform applied to hidden states.
        bias (`bool`, *optional*, defaults to True):
            Whether to apply bias to the final prediction layer.
        act (`str`, *optional*, defaults to None):
            The activation function of the final prediction output.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """

    hidden_size: Optional[int] = None
    dropout: float = 0.0
    transform: Optional[str] = "nonlinear"
    transform_act: Optional[str] = "gelu"
    bias: bool = True
    act: Optional[str] = None
    layer_norm_eps: float = 1e-12
