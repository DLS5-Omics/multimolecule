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
    AutoModelForMethylationPrediction,
    AutoModelForPolyadenylationPrediction,
    AutoModelForRegulatoryActivityPrediction,
    AutoModelForRegulatoryProfilePrediction,
    AutoModelForRegulatoryTrackPrediction,
    AutoModelForRegulatoryVariantEffectPrediction,
    AutoModelForRnaSecondaryStructurePrediction,
    AutoModelForSpliceSitePrediction,
    AutoModelForSpliceVariantEffectPrediction,
)
from .methylation import MethylationPipeline
from .polyadenylation import PolyadenylationPipeline
from .regulatory import (
    RegulatoryActivityPipeline,
    RegulatoryProfilePipeline,
    RegulatoryTrackPipeline,
    RegulatoryVariantEffectPipeline,
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

PIPELINE_REGISTRY.register_pipeline(
    "polyadenylation",
    PolyadenylationPipeline,
    pt_model=AutoModelForPolyadenylationPrediction,
    default={"model": ("multimolecule/aparent2", "main")},
)

PIPELINE_REGISTRY.register_pipeline(
    "regulatory-activity",
    RegulatoryActivityPipeline,
    pt_model=AutoModelForRegulatoryActivityPrediction,
    default={"model": ("multimolecule/basset", "main")},
)

PIPELINE_REGISTRY.register_pipeline(
    "regulatory-track",
    RegulatoryTrackPipeline,
    pt_model=AutoModelForRegulatoryTrackPrediction,
    default={"model": ("multimolecule/enformer", "main")},
)

PIPELINE_REGISTRY.register_pipeline(
    "regulatory-profile",
    RegulatoryProfilePipeline,
    pt_model=AutoModelForRegulatoryProfilePrediction,
    default={"model": ("multimolecule/bpnet", "main")},
)

PIPELINE_REGISTRY.register_pipeline(
    "regulatory-variant-effect",
    RegulatoryVariantEffectPipeline,
    pt_model=AutoModelForRegulatoryVariantEffectPrediction,
    default={"model": ("multimolecule/deepsea", "main")},
)

PIPELINE_REGISTRY.register_pipeline(
    "methylation",
    MethylationPipeline,
    pt_model=AutoModelForMethylationPrediction,
    default={"model": ("multimolecule/deepcpgdna-smallwood2014-serum", "main")},
)

__all__ = [
    "RnaSecondaryStructurePipeline",
    "SpliceSitePipeline",
    "SpliceVariantEffectPipeline",
    "PolyadenylationPipeline",
    "RegulatoryActivityPipeline",
    "RegulatoryTrackPipeline",
    "RegulatoryProfilePipeline",
    "RegulatoryVariantEffectPipeline",
    "MethylationPipeline",
]
