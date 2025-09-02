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


from multimolecule.utils.rna.secondary_structure import bprna, topology


def test_bprna_empty_structure() -> None:
    structure = topology.RnaSecondaryStructure("", "")
    assert bprna.annotate_structure(structure) == ""
    assert bprna.annotate_function(structure) == ""


def test_bprna_hairpin_annotation() -> None:
    structure = topology.RnaSecondaryStructure("ACG", "(.)")
    assert bprna.annotate_structure(structure) == "SHS"
    assert bprna.annotate_function(structure) == "NNN"


def test_bprna_pseudoknot_annotation() -> None:
    structure = topology.RnaSecondaryStructure("ACGU", "([)]")
    structural_annotation, functional_annotation = bprna.annotate(structure)
    assert structural_annotation == "SPPS"
    assert functional_annotation == "NKNK"
