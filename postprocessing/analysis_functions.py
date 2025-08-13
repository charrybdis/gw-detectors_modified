import numpy as np

def merge_dicts(dict_list):
    """
    Merges the values corresponding to a key in a list of dictionaries.
    """
    result = {}
    for d in dict_list:
        for key, value in d.items():
            if key in result:
                result[key].append(value)
            else:
                result[key] = [value]
    return result


def reshape_results(list_results, num, truncate=True):
    """
    Rearranges results produced using truncated version of az_po_meshgrid to match a typical grid of coordinates. 
    Assumes the missing coordinates (azimuth, 0) and (azimuth, pi) were added to the end of the list of coordinates 
    in that order, and that meshgrid is indexed with 'ij'.
    
    Parameters: 
    num --- (Int) Resolution given to az_po_meshgrid.
    list_results --- (List) List of values to rearrange.
    """
    if truncate==True:
        # reshapes so azimuths correspond to rows, poles to columns
        reshaped = np.reshape(list_results[:-2], (num-1, num-2))

        # recovers values corresponding to (azimuth, 0) and (azimuth, pi)
        az_zero = list_results[-2]
        az_pi = list_results[-1]
        az_pi_arr = np.full((num-1,), az_pi) # populates column with that value
        az_zero_arr = np.full((num-1,), az_zero)

        # stacks columns on either side of reshaped array
        poles_added = np.column_stack((az_pi_arr, reshaped, az_zero_arr))

        # recovers values for (-pi, poles)
        pi_x_arr = poles_added[-1,:]

        # stacks row on top of new array
        grid = np.row_stack((pi_x_arr, poles_added))
    else:
        grid = np.reshape(list_results, (num,num))
    
    return grid


def approx_bayes_factor(true_snr, filter_response):
    """
    Approximates the Bayes factor with the maximum likelihood estimates of the observed and optimal snr. 
    
    Parameters:
    true_snr --- (Array or Number) True (optimal) SNR. 
    filter_response --- (Array or Number) Filter (observed) SNR.
    """
    match = filter_response / true_snr
    return np.exp(true_snr**2 * (match - 1) / 2)


def approx_threshold(B, match):
    """
    Approximates optimal SNR needed to obtain the desired Bayes factor given a particular match value.
    
    Parameters:
    B --- (Number) Desired Bayes factor.
    match --- (Number) filter / snr. 
    """
    return np.sqrt(2 * np.log(B) / np.abs(match**2 - 1))


def standard_threshold(match):
    """
    Assumes |m^2 - 1| ~ 1 / rho_opt^2.
    """
    return np.sqrt(1 / np.abs(match**2 - 1))