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

from functools import partial
from typing import Type

from danling.metric import Metrics
from danling.metric.functional import (
    binary_accuracy,
    binary_auprc,
    binary_auroc,
    binary_f1_score,
    mcc,
    multiclass_accuracy,
    multiclass_auprc,
    multiclass_auroc,
    multiclass_f1_score,
    multilabel_accuracy,
    multilabel_auprc,
    multilabel_auroc,
    multilabel_f1_score,
    pearson,
    r2_score,
    rmse,
    spearman,
)
from danling.metric.preprocess import (
    preprocess_binary,
    preprocess_multiclass,
    preprocess_multilabel,
    preprocess_regression,
)


def binary_metrics(ignore_index: int | None = -100, cls: Type = Metrics, **kwargs):
    """
    Create a pre-configured Metrics instance for binary classification tasks.

    This factory function returns a Metrics object with a standard set of binary
    classification metrics, including:
    - AUROC (Area Under ROC Curve)
    - AUPRC (Area Under Precision-Recall Curve)
    - Accuracy
    - F1 Score
    - MCC (Matthews Correlation Coefficient)

    The returned Metrics instance is ready to use with model predictions/logits
    and binary labels (0/1 or False/True).

    Args:
        ignore_index: Value in the target to ignore (e.g., padding).
        **kwargs: Additional metric functions to include or override default metrics

    Returns:
        Metrics: A configured Metrics instance for binary classification
    """
    return cls(
        auroc=binary_auroc,
        auprc=binary_auprc,
        acc=binary_accuracy,
        f1=binary_f1_score,
        mcc=mcc,
        preprocess=partial(preprocess_binary, ignore_index=ignore_index),
        **kwargs,
    )


def multiclass_metrics(
    num_classes: int, average: str = "macro", ignore_index: int | None = -100, cls: Type = Metrics, **kwargs
):
    """
    Create a pre-configured Metrics instance for multiclass classification tasks.

    This factory function returns a Metrics object with a standard set of multiclass
    classification metrics, including:
    - AUROC
    - AUPRC
    - Accuracy
    - F1 Score
    - MCC (Matthews Correlation Coefficient)

    The returned Metrics instance is ready to use with model logits (shape [batch_size, num_classes])
    and class labels (shape [batch_size]).

    Args:
        num_classes: Number of classes in the classification task
        ignore_index: Value in the target to ignore (e.g., padding).
        **kwargs: Additional metric functions to include or override default metrics

    Returns:
        Metrics: A configured Metrics instance for multiclass classification
    """
    return cls(
        auroc=partial(multiclass_auroc, num_classes=num_classes, average=average),
        auprc=partial(multiclass_auprc, num_classes=num_classes, average=average),
        acc=partial(multiclass_accuracy, num_classes=num_classes, average=average),
        f1=partial(multiclass_f1_score, num_classes=num_classes, average=average),
        mcc=partial(mcc, task="multiclass", num_classes=num_classes),
        preprocess=partial(preprocess_multiclass, num_classes=num_classes, ignore_index=ignore_index),
        **kwargs,
    )


def multilabel_metrics(
    num_labels: int, average: str = "macro", ignore_index: int | None = -100, cls: Type = Metrics, **kwargs
):
    """
    Create a pre-configured Metrics instance for multi-label classification tasks.

    In multi-label classification, each sample can belong to multiple classes simultaneously.
    This factory returns a Metrics object with metrics appropriate for multi-label tasks:
    - AUROC
    - AUPRC
    - Accuracy
    - F1 Score
    - MCC (Matthews Correlation Coefficient, per label)

    The returned Metrics instance expects model outputs with shape [batch_size, num_labels]
    and binary labels with the same shape.

    Args:
        num_labels: Number of possible labels in the multi-label task
        ignore_index: Value in the target to ignore.
        **kwargs: Additional metric functions to include or override default metrics

    Returns:
        Metrics: A configured Metrics instance for multi-label classification
    """
    return cls(
        auroc=partial(multilabel_auroc, num_labels=num_labels, average=average),
        auprc=partial(multilabel_auprc, num_labels=num_labels, average=average),
        acc=partial(multilabel_accuracy, num_labels=num_labels),
        f1=partial(multilabel_f1_score, num_labels=num_labels, average=average),
        mcc=partial(mcc, task="multilabel", num_labels=num_labels),
        preprocess=partial(preprocess_multilabel, num_labels=num_labels, ignore_index=ignore_index),
        **kwargs,
    )


def regression_metrics(num_outputs: int = 1, ignore_nan: bool = True, cls: Type = Metrics, **kwargs):
    """
    Create a pre-configured Metrics instance for regression tasks.

    This factory function returns a Metrics object with standard regression metrics:
    - R² Score
    - Pearson Correlation Coefficient
    - Spearman Correlation Coefficient
    - RMSE

    The metrics can handle both single-output regression (e.g., predicting a single value)
    and multi-output regression (e.g., predicting multiple values per sample).

    Args:
        num_outputs: Number of regression outputs per sample.
        ignore_nan: Whether to ignore NaN values in inputs/targets.
        **kwargs: Additional metric functions to include or override default metrics

    Returns:
        Metrics: A configured Metrics instance for regression tasks
    """
    return cls(
        pearson=partial(pearson, num_outputs=num_outputs),
        spearman=partial(spearman, num_outputs=num_outputs),
        r2=partial(r2_score, num_outputs=num_outputs),
        rmse=partial(rmse, num_outputs=num_outputs),
        preprocess=partial(preprocess_regression, num_outputs=num_outputs, ignore_nan=ignore_nan),
        **kwargs,
    )
