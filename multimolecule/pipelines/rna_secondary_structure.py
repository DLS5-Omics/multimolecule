# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

from typing import Dict
from warnings import warn

from transformers.pipelines.base import GenericTensor, Pipeline, PipelineException

from ..data.functional import contact_map_to_dot_bracket


class RnaSecondaryStructurePipeline(Pipeline):
    """
    RNA secondary structure prediction pipeline using any `ModelWithSecondaryStructureHead`.

    Examples:

    ```python
    >>> import multimolecule
    >>> from transformers import pipeline

    >>> predictor = pipeline("rna-secondary-structure")
    >>> output = predictor("UAGCUUAUCAGACUGAUGUUG")
    >>> output["secondary_structure"]
    '.....................'

    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This secondary structure prediction pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"rna-secondary-structure"`.

    The models that this pipeline can use are models that have been trained with a RNA secondary structure prediction
    objective, which includes the bi-directional models in the library. See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=rna-secondary-structure).
    """

    threshold: float = 0.5
    output_contact_map: bool = False

    def preprocess(
        self, inputs, return_tensors=None, tokenizer_kwargs=None, **preprocess_parameters
    ) -> Dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = self.framework
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        model_outputs["input_ids"] = model_inputs["input_ids"]

        if len(model_inputs["input_ids"]) > 1 and getattr(self.model, "supports_batch_process", False):
            warn(
                "The pipeline received a batch of sequences as input.\n"
                "Most RNA Secondary Structure models are designed and trained with a single sequence.\n"
                "The results may be less reliable in batch processing.\n",
                RuntimeWarning,
            )
        return model_outputs

    def postprocess(self, model_outputs, threshold: float | None = None, output_contact_map: bool | None = None):
        if threshold is None:
            threshold = self.threshold
        if output_contact_map is None:
            output_contact_map = self.output_contact_map

        input_ids = model_outputs["input_ids"]
        if hasattr(self.model, "postprocess"):
            outputs = self.model.postprocess(model_outputs)
        else:
            if "logits_ss" in model_outputs:
                outputs = model_outputs["logits_ss"]
            elif "logits" in model_outputs:
                outputs = model_outputs["logits"]
            else:
                raise PipelineException(
                    "rna-secondary-structure", self.model.base_model_prefix, "Unable to find logits in model outputs."
                )
            outputs = outputs.squeeze(-1)
            outputs = outputs.sigmoid()

        if len(input_ids) == 1:
            input_ids = input_ids.squeeze(0)
            outputs = outputs.squeeze(0)
            sequence = self.tokenizer.decode(input_ids, skip_special_tokens=True).replace(" ", "")
            dot_bracket = contact_map_to_dot_bracket(outputs > threshold, unsafe=True)
            ret = {"sequence": sequence, "secondary_structure": dot_bracket}
            if output_contact_map:
                ret["contact_map"] = outputs.tolist()
            return ret

        sequences = [i.replace(" ", "") for i in self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)]
        dot_brackets = []
        if self.output_contact_map:
            contact_maps = []
        for sequence, contact_map in zip(sequences, outputs):
            contact_map = contact_map[: len(sequence), : len(sequence)]
            dot_brackets.append(contact_map_to_dot_bracket(contact_map > threshold, unsafe=True))
            if self.output_contact_map:
                contact_maps.append(contact_map.tolist())

        ret = {"sequence": sequences, "secondary_structure": dot_brackets}
        if self.output_contact_map:
            ret["contact_map"] = contact_maps

        return ret

    def _sanitize_parameters(
        self, threshold: float | None = None, output_contact_map: bool | None = None, tokenizer_kwargs=None
    ):
        preprocess_params = {}

        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs

        postprocess_params = {}

        if threshold is not None:
            postprocess_params["threshold"] = threshold
            if threshold >= 1:
                raise PipelineException(
                    "rna-secondary-structure",
                    self.model.base_model_prefix,
                    f"Threshold must be less than 1, but got {threshold}.",
                )
            if threshold <= 0:
                raise PipelineException(
                    "rna-secondary-structure",
                    self.model.base_model_prefix,
                    f"Threshold must be greater than 0, but got {threshold}.",
                )
        if output_contact_map is not None:
            if not isinstance(output_contact_map, bool):
                raise PipelineException(
                    "rna-secondary-structure",
                    self.model.base_model_prefix,
                    f"output_contact_map must be a boolean, but got {type(output_contact_map)}.",
                )
            postprocess_params["output_contact_map"] = output_contact_map
        return preprocess_params, {}, postprocess_params

    def __init__(self, *args, threshold: float | None = None, output_contact_map: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.framework != "pt":
            raise NotImplementedError("Only PyTorch is supported for RNA secondary structure prediction.")
        if threshold is not None:
            if threshold >= 1:
                raise PipelineException(
                    "rna-secondary-structure",
                    self.model.base_model_prefix,
                    f"Threshold must be less than 1, but got {threshold}.",
                )
            if threshold <= 0:
                raise PipelineException(
                    "rna-secondary-structure",
                    self.model.base_model_prefix,
                    f"Threshold must be greater than 0, but got {threshold}.",
                )
            self.threshold = threshold
        if output_contact_map is not None:
            if not isinstance(output_contact_map, bool):
                raise PipelineException(
                    "rna-secondary-structure",
                    self.model.base_model_prefix,
                    f"output_contact_map must be a boolean, but got {type(output_contact_map)}.",
                )
            self.output_contact_map = output_contact_map

    def __call__(self, inputs, **kwargs):
        """
        Predict the secondary structure of the RNA sequence(s) given as inputs.

        Args:
            inputs (`str` or `List[str]`):
                One or several RNA sequences.
            threshold (`float`, *optional*):
                The threshold to use for determining if a contact is present or not. If not provided, the default is
                `0.5`. The value must be between 0 and 1.
            output_contact_map (`bool`, *optional*):
                Whether to output the contact map along with the secondary structure. If not provided, the default is
                `False`.

        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:

            - **sequence** (`str`) -- The corresponding input with the mask token prediction.
            - **score** (`float`) -- The corresponding probability.
            - **token** (`int`) -- The predicted token id (to replace the masked one).
            - **token_str** (`str`) -- The predicted token (to replace the masked one).
        """
        outputs = super().__call__(inputs, **kwargs)
        if isinstance(inputs, list) and len(inputs) == 1:
            return outputs[0]
        return outputs
