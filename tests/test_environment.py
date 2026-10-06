def test_environment():
    import pandas
    import numpy
    import sklearn
    import matplotlib

    assert pandas is not None
    assert numpy is not None
    assert sklearn is not None
    assert matplotlib is not None