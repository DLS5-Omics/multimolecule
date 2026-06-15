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
# along with MultiMolecule.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

import pytest

from multimolecule import io


def test_read_ct_reads_mfold_style(tmp_path) -> None:
    path = tmp_path / "mfold.ct"
    path.write_text(
        "  4  ENERGY = -1.2  myrna\n"
        "  1 A    0    2    0    1\n"
        "  2 C    1    3    4    2\n"
        "  3 G    2    4    0    3\n"
        "  4 U    3    0    2    4\n"
    )

    record = io.read_ct(path)

    assert record.sequence == "ACGU"
    assert record.dot_bracket == ".(.)"
    assert record.id == "myrna"


def test_write_ct_golden(tmp_path) -> None:
    record = io.RnaSecondaryStructureRecord(sequence="GGGAAACCC", dot_bracket="(((...)))", id="hp")
    path = tmp_path / "hp.ct"

    io.write_ct(record, path)

    expected = (
        "9 hp\n"
        "1 G 0 2 9 1\n"
        "2 G 1 3 8 2\n"
        "3 G 2 4 7 3\n"
        "4 A 3 5 0 4\n"
        "5 A 4 6 0 5\n"
        "6 A 5 7 0 6\n"
        "7 C 6 8 3 7\n"
        "8 C 7 9 2 8\n"
        "9 C 8 0 1 9\n"
    )
    assert path.read_text() == expected


def test_read_ct_rejects_header_count_mismatch(tmp_path) -> None:
    path = tmp_path / "bad.ct"
    path.write_text("3 title\n1 A 0 2 0 1\n2 C 1 0 0 2\n")

    with pytest.raises(io.InvalidStructureFile):
        io.read_ct(path)
