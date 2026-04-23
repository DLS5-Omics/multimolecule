# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule
#
# This file is part of MultiMolecule.
#
# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

from multimolecule.metrics.rna.secondary_structure import paired
from tests.metrics.rna.secondary_structure.utils import assert_confusion_metric_family


def test_paired_nucleotide_metric_family(perfect_nested_context) -> None:
    assert_confusion_metric_family(
        perfect_nested_context,
        paired.paired_nucleotides_confusion,
        paired.paired_nucleotides_precision,
        paired.paired_nucleotides_recall,
        paired.paired_nucleotides_f1,
    )


def test_paired_nucleotide_mcc_is_bounded(perfect_nested_context) -> None:
    value = paired.paired_nucleotides_mcc(perfect_nested_context)
    assert -1.0 <= float(value.item()) <= 1.0
