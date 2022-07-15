def calc_error(actual: float, expected: float) -> float:
    """
    Calculate the error between two numbers.

    :param actual: The actual value.
    :type actual: float
    :param expected: The expected value.
    :type expected: float
    :return: The error in absolute value (>=0).
    :rtype: float
    """
    if actual == 0.0:
        return abs(actual)
    return abs(actual - expected) / expected
