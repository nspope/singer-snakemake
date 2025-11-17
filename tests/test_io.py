import pytest
import numpy as np
from io import StringIO

from workflow.scripts.utils import read_single_fasta
from workflow.scripts.utils import write_minimal_vcf


def test_read_single_fasta():
    fasta = ">foo\nbar\nbaz"
    fasta_body = read_single_fasta(StringIO(fasta))
    assert np.all(fasta_body == np.array(["b", "a", "r", "b", "a", "z"]))
    fasta_body = read_single_fasta(StringIO(fasta + "\n"))
    assert np.all(fasta_body == np.array(["b", "a", "r", "b", "a", "z"]))
    with pytest.raises(AssertionError):
        read_single_fasta(StringIO(fasta + "\n>boo\nbaz\n"))
    with pytest.raises(AssertionError):
        read_single_fasta(StringIO("foo\nbar"))
    with pytest.raises(AssertionError):
        read_single_fasta(StringIO("foo"))


def test_write_minimal_vcf():
    # TODO
    pass
