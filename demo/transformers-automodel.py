# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import multimolecule  # noqa: F401

model = AutoModelForSequenceClassification.from_pretrained("multimolecule/mrnafm")
tokenizer = AutoTokenizer.from_pretrained("multimolecule/mrnafm")

sequence = "UAGCGUAUCAGACUGAUGUUG"
output = model(**tokenizer(sequence, return_tensors="pt"))
