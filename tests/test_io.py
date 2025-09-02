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


from multimolecule import io


def test_dbn_roundtrip(tmp_path) -> None:
    record = io.RnaSecondaryStructureRecord(sequence="ACGU", dot_bracket="(())", id="test")
    path = tmp_path / "test.dbn"
    io.write_dbn(record, path)
    out = io.read_dbn(path)
    assert out.sequence == record.sequence
    assert out.dot_bracket == record.dot_bracket
    assert out.id == record.id


def test_dbn_energy_line(tmp_path) -> None:
    path = tmp_path / "energy.dbn"
    path.write_text(">id\nACGU\n(()) -3.4\n")
    record = io.read_dbn(path)
    assert record.sequence == "ACGU"
    assert record.dot_bracket == "(())"
    assert record.id == "id"


def test_bpseq_roundtrip(tmp_path) -> None:
    record = io.RnaSecondaryStructureRecord(sequence="ACGU", dot_bracket="(())")
    path = tmp_path / "example.bpseq"
    io.write_bpseq(record, path)
    out = io.read_bpseq(path)
    assert out.sequence == record.sequence
    assert out.dot_bracket == record.dot_bracket


def test_fasta_roundtrip(tmp_path) -> None:
    record = io.SequenceRecord(sequence="ACGU", id="rna1", comment="note")
    path = tmp_path / "example.fasta"
    io.write_fasta(record, path)
    out = io.read_fasta(path)
    assert out.sequence == record.sequence
    assert out.id == record.id
    assert out.comment == record.comment


def test_fasta_plain_sequence(tmp_path) -> None:
    path = tmp_path / "plain.fa"
    path.write_text(">id some description\nACGU\n")
    out = io.read_fasta(path)
    assert out.sequence == "ACGU"
    assert out.id == "id"
    assert out.comment == "some description"


def test_st_roundtrip(tmp_path) -> None:
    record = io.BpRnaRecord(
        sequence="ACGU",
        dot_bracket="(())",
        id="rna1",
        page_number=1,
        structure_array="SSSS",
        knot_array="NNNN",
        structure_types={"S": ["S1 1..2 3..4"]},
    )
    path = tmp_path / "example.st"
    io.write_st(record, path)
    out = io.read_st(path)
    assert out.id == "rna1"
    assert out.sequence == "ACGU"
    assert out.dot_bracket == "(())"
    assert out.structure_array == "SSSS"
    assert out.knot_array == "NNNN"
    assert out.structure_types["S"][0].strip() == "S1 1..2 3..4"


def test_sta_multiple_records(tmp_path) -> None:
    record = io.BpRnaRecord(
        sequence="ACGU",
        dot_bracket="(())",
        id="rna1",
        page_number=1,
        structure_array="SSSS",
        knot_array="NNNN",
        structure_types={},
    )
    path = tmp_path / "single.st"
    io.write_st(record, path)
    out = io.read_st(path)
    assert out.id == "rna1"
    assert out.dot_bracket == "(())"
