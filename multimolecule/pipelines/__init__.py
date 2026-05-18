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


from transformers.pipelines import PIPELINE_REGISTRY

from ..models.modeling_auto import (
    AutoModelForRnaSecondaryStructurePrediction,
    AutoModelForSpliceSitePrediction,
    AutoModelForSpliceVariantEffectPrediction,
)
from .rna_secondary_structure import RnaSecondaryStructurePipeline
from .splicing import SpliceSitePipeline, SpliceVariantEffectPipeline

PIPELINE_REGISTRY.register_pipeline(
    "rna-secondary-structure",
    RnaSecondaryStructurePipeline,
    pt_model=AutoModelForRnaSecondaryStructurePrediction,
    default={"model": ("multimolecule/ernierna-ss", "main")},
)

PIPELINE_REGISTRY.register_pipeline(
    "splice-site",
    SpliceSitePipeline,
    pt_model=AutoModelForSpliceSitePrediction,
    default={"model": ("multimolecule/openspliceai-mane.400", "main")},
)

PIPELINE_REGISTRY.register_pipeline(
    "splice-variant-effect",
    SpliceVariantEffectPipeline,
    pt_model=AutoModelForSpliceVariantEffectPrediction,
    default={"model": ("multimolecule/mmsplice", "main")},
)

__all__ = ["RnaSecondaryStructurePipeline", "SpliceSitePipeline", "SpliceVariantEffectPipeline"]
