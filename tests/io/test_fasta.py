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


def test_fasta_multi_records(tmp_path) -> None:
    path = tmp_path / "multi.fasta"
    path.write_text(">rna1 first\nACGU\n>rna2\nUGCA\n")
    records = io.read_fasta_records(path)
    assert [record.id for record in records] == ["rna1", "rna2"]
    assert [record.sequence for record in records] == ["ACGU", "UGCA"]
    assert records[0].comment == "first"
    assert io.read_fasta(path).id == "rna1"

    out_path = tmp_path / "multi_out.fasta"
    io.write_fasta_records(records, out_path)
    out = io.read_fasta_records(out_path)
    assert [record.id for record in out] == ["rna1", "rna2"]
    assert [record.sequence for record in out] == ["ACGU", "UGCA"]


def test_fasta_extension_aliases(tmp_path) -> None:
    record = io.SequenceRecord(sequence="ACGU", id="rna1")
    path = tmp_path / "example.ffn"
    io.save(record, path)
    out = io.load(path)
    assert out.sequence == record.sequence
    assert out.id == record.id
