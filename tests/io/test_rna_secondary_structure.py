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

import pytest

from multimolecule import io


@pytest.mark.parametrize("fmt", ["dbn", "bpseq", "ct"])
@pytest.mark.parametrize(
    "sequence, dot_bracket",
    [
        ("ACGU", "(())"),  # nested
        ("ACGU", "...."),  # fully unpaired
        ("ACGU", "([)]"),  # pseudoknot: crossing pairs survive the pairing table
        ("GGGAAACCC", "(((...)))"),  # hairpin
    ],
)
def test_roundtrip(tmp_path, fmt, sequence, dot_bracket) -> None:
    record = io.RnaSecondaryStructureRecord(sequence=sequence, dot_bracket=dot_bracket, id="rna")
    path = tmp_path / f"rna.{fmt}"
    io.save(record, path)
    out = io.load(path)
    assert out.sequence == sequence
    assert out.dot_bracket == dot_bracket
    assert out.id == "rna"


def test_dbn_energy_line(tmp_path) -> None:
    path = tmp_path / "energy.dbn"
    path.write_text(">id\nACGU\n(()) -3.4\n")
    record = io.read_dbn(path)
    assert record.sequence == "ACGU"
    assert record.dot_bracket == "(())"
    assert record.id == "id"


@pytest.fixture
def st_record() -> "io.BpRnaRecord":
    return io.BpRnaRecord(
        sequence="ACGU",
        dot_bracket="(())",
        id="rna1",
        page_number=1,
        structure_array="SSSS",
        knot_array="NNNN",
        structure_types={"S": ["S1 1..2 3..4"]},
    )


def test_st_roundtrip(tmp_path, st_record) -> None:
    path = tmp_path / "example.st"
    io.write_st(st_record, path)
    out = io.read_st(path)
    assert out.id == "rna1"
    assert out.sequence == "ACGU"
    assert out.dot_bracket == "(())"
    assert out.structure_array == "SSSS"
    assert out.knot_array == "NNNN"
    assert out.structure_types["S"][0].strip() == "S1 1..2 3..4"


def test_st_comma_wrapped_length_header(tmp_path) -> None:
    path = tmp_path / "comma_length.st"
    path.write_text("#Name: rna1\n#Length: ,4,\n#PageNumber: 1\nACGU\n(())\nSSSS\nNNNN\n")
    out = io.read_st(path)
    assert out.id == "rna1"
    assert out.sequence == "ACGU"
    assert out.dot_bracket == "(())"
