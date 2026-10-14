import numpy as np

from ratings.xo import def_group, off_code, parse_personnel, shell


def test_personnel_both_formats():
    old = parse_personnel("1 RB, 2 TE, 2 WR")
    new = parse_personnel("1 C, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 2 WR")
    assert off_code(old) == off_code(new) == "12"
    assert off_code(parse_personnel("2 RB, 1 TE, 2 WR")) == "21"
    assert off_code({}) == "unknown"


def test_defense_groups_both_formats():
    assert def_group(parse_personnel("4 DL, 2 LB, 5 DB")) == "nickel"
    assert def_group(parse_personnel("4 DL, 3 LB, 4 DB")) == "base"
    assert def_group(parse_personnel("3 CB, 2 DE, 2 DT, 1 FS, 2 ILB, 1 SS")) == "nickel"
    assert def_group(parse_personnel("4 CB, 2 DE, 1 DT, 1 FS, 1 ILB, 1 SS")) == "dime"


def test_shells():
    assert shell("COVER_3") == "single-high" and shell("COVER_4") == "two-high"
    assert shell("COVER_0") == "cover0" and shell("BLOWN") == "other"
    assert np.isnan(shell(None)) and np.isnan(shell(""))
