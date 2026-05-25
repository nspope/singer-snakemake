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


def test_write_minimal_vcf_diploid():
    sample_names = np.array(["sA", "sB", "sC"])
    CHROM = np.array(["chr1", "chr1", "chr1"])
    POS = np.array([100, 200, 300], dtype=np.int64)
    ID = np.array(["v0", "v1", "v2"])
    REF = np.array(["A", "C", "G"])
    ALT = np.array(["T", "G", "A"])
    GT = np.array([
        [[ 0, 1], [1,  0], [ 0,  0]],
        [[ 1, 1], [0,  0], [ 1,  1]],
        [[-1, 0], [1, -1], [-1, -1]],
    ]).astype(np.int8)
    # diploid 
    buffer = StringIO()
    write_minimal_vcf(buffer, sample_names, CHROM, POS, ID, REF, ALT, GT, ploidy=2)
    lines = buffer.getvalue().splitlines()
    assert lines[0] == "##fileformat=VCFv4.2"
    assert lines[1].startswith("##source=")
    assert lines[2] == '##FILTER=<ID=PASS,Description="All filters passed">'
    assert lines[3] == '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">'
    cols = lines[4].split("\t")
    assert cols[:9] == ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    assert cols[9:] == ["sA", "sB", "sC"]
    row0 = lines[5].split("\t")
    assert row0[:9] == ["chr1", "100", "v0", "A", "T", ".", "PASS", ".", "GT"]
    assert row0[9:] == ["0|1", "1|0", "0|0"]
    row1 = lines[6].split("\t")
    assert row1[:5] == ["chr1", "200", "v1", "C", "G"]
    assert row1[9:] == ["1|1", "0|0", "1|1"]
    row2 = lines[7].split("\t")
    assert row2[:5] == ["chr1", "300", "v2", "G", "A"]
    assert row2[9:] == [".|0", "1|.", ".|."]
    assert len(lines) == 8
    # haploid
    sample_names = np.array(["sA", "sB", "sC", "sD", "sE", "sF"])
    GT = np.concatenate([GT[..., 0], GT[..., 1]], axis=1)
    GT = np.stack([GT, -np.ones_like(GT)], axis=-1)
    buffer = StringIO()
    write_minimal_vcf(buffer, sample_names, CHROM, POS, ID, REF, ALT, GT, ploidy=1)
    lines = buffer.getvalue().splitlines()
    assert lines[0] == "##fileformat=VCFv4.2"
    assert lines[1].startswith("##source=")
    assert lines[2] == '##FILTER=<ID=PASS,Description="All filters passed">'
    assert lines[3] == '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">'
    cols = lines[4].split("\t")
    assert cols[:9] == ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    assert cols[9:] == ["sA", "sB", "sC", "sD", "sE", "sF"]
    row0 = lines[5].split("\t")
    assert row0[:9] == ["chr1", "100", "v0", "A", "T", ".", "PASS", ".", "GT"]
    assert row0[9:] == ["0", "1", "0", "1", "0", "0"]
    row1 = lines[6].split("\t")
    assert row1[:5] == ["chr1", "200", "v1", "C", "G"]
    assert row1[9:] == ["1", "0", "1", "1", "0", "1"]
    row2 = lines[7].split("\t")
    assert row2[:5] == ["chr1", "300", "v2", "G", "A"]
    assert row2[9:] == [".", "1", ".", "0", ".", "."]
    assert len(lines) == 8


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

