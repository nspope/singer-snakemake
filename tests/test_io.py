import pytest
import numpy as np
from io import StringIO

from workflow.scripts.utils import read_single_fasta
from workflow.scripts.utils import write_minimal_vcf
from workflow.scripts.utils import parse_sample_bedmask


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


@pytest.mark.skip("TODO")
def test_write_minimal_vcf():
    assert False


def test_parse_sample_bedmask(tmp_path):
    bed_path = tmp_path / "sample_intervals.bed"
    bed_content = (
        "chr10\t100\t200\tsampleA\n"
        "chr10\t150\t300\tsampleA\n"     # overlaps with previous
        "chr10\t500\t600\tsampleA\n"
        "chr10\t50\t80\tsampleB\n"
        "chr10\t90\t120\tsampleB\n"      # gap between 80-90
        "chr10\t1000\t2000\tsampleC\n"   # not in sample_map
        "chr10\t10\t20\tsampleA\n"       # out of order
    )
    with open(bed_path, "w") as handle:
        handle.write(bed_content)

    sample_map = {"sampleA": 0, "sampleB": 1}
    intervals_by_sample = parse_sample_bedmask(bed_path, sample_map)

    assert 0 in intervals_by_sample
    sampleA = intervals_by_sample[0].tolist()
    assert sampleA == [[10, 20], [100, 300], [500, 600]]

    assert 1 in intervals_by_sample
    sampleB = intervals_by_sample[1].tolist()
    assert sampleB == [[50, 80], [90, 120]]

    assert 2 not in intervals_by_sample  # not in sample_map

