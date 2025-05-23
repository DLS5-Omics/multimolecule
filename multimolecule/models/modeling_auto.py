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


from collections import OrderedDict

from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


class AutoModelForSequencePrediction(_BaseAutoModelClass):
    _model_mapping = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())


class AutoModelForTokenPrediction(_BaseAutoModelClass):
    _model_mapping = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())


class AutoModelForContactPrediction(_BaseAutoModelClass):
    _model_mapping = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())


class AutoModelForRnaSecondaryStructurePrediction(_BaseAutoModelClass):
    _model_mapping = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
