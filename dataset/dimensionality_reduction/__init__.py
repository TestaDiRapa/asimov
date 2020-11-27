import math
import numpy as np


def magnitude_order(x):
    """
    Returns the magnitude order of a number
    :param x: a number
    :return: an int
    """
    return int(math.log10(x))


def coefficients_by_magnitude(coefficients, omic_array):
    """
    Given a set of coefficients and an omic array, returns a dictionary where each key is an order of magnitude and
    each value is another dictionary that contains the features for each order of magnitude of the coefficients.
    :param coefficients: an array of coefficients
    :param omic_array: an omic array. The number of features of the omic part must be the same length of coefficients
    :return: a dictionary
    """
    results = dict()
    start = magnitude_order(min(np.abs(coefficients)))
    stop = magnitude_order(max(np.abs(coefficients)))
    results[start] = {
        "feats": omic_array.get_omic_column_index().to_series().loc[np.abs(coefficients) <= 10**start].to_list(),
        "coefficients": coefficients[np.abs(coefficients) <= 10**start]
    }
    c = np.count_nonzero(np.abs(coefficients) <= 10**start)
    for magnitude in range(start+1, stop+1):
        condition = (10**(magnitude-1) < np.abs(coefficients)) & (np.abs(coefficients) <= 10**magnitude)
        c += np.count_nonzero(condition)
        results[magnitude] = {
            "feats": omic_array.get_omic_column_index().to_series().loc[condition].to_list(),
            "coefficients": coefficients[condition]
        }
    return results
