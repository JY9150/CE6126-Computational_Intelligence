from MyLib.Old import Old


def test_Old():
    old_man = Old("Mark Muller")
    assert old_man.name == "Mark Muller"