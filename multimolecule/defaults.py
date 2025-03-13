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

from datasets.data_files import SPLIT_KEYWORDS, Split

TRAIN_SPLITS = tuple(SPLIT_KEYWORDS[Split.TRAIN])
VALIDATION_SPLITS = tuple(SPLIT_KEYWORDS[Split.VALIDATION])
TEST_SPLITS = tuple(SPLIT_KEYWORDS[Split.TEST])
INFERENCE_SPLITS = ("inf", "inference")
DATASET_SPLITS = TRAIN_SPLITS + VALIDATION_SPLITS + TEST_SPLITS + INFERENCE_SPLITS
ID_COL_NAMES = ["id", "idx", "index"]
SEQUENCE_COL_NAMES = ["input_ids", "sequence", "seq"]
SECONDARY_STRUCTURE_COL_NAMES = ["secondary_structure", "ss"]
LABEL_COL_NAMES = ["label", "labels"]
SEQUENCE_COL_NAME = "sequence"
LABEL_COL_NAME = "labels"
LABLE_TYPE_THRESHOLD = 0.5
TASK_INFERENCE_NUM_ROWS = 100
