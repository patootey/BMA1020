"""
This code uses hermite interpolation to convert RGB to LMS and viseversa.
Got from assignment 8.

@Author Mathilde Borsodi Hegna
"""

import numpy as np


def closest(target_time, time_points):
    """
    Find the index of the interval [t_i, t_{i+1}) that contains the target_time.

    :param target_time: The time to evaluate.
    :param time_pints: Ordered list of unique time points.
    :return: Index of the interval or -1 if target_time is out of bounds.
    """
    # Check if target_time is within the range of time
    if target_time < time_points[0] or target_time > time_points[-1]:
        return -1

    for index in range(len(time_points) - 1):
        if time_points[index] <= target_time < time_points[index + 1]:
            return index

    return -1  # default value if no interval is found


def hermite_interpolation(target_time, time, pos, speed):
    """
    Perform Hermite interpolation to compute position and velocity at a target time.

    :param target_time: Time at which to evaluate interpolation.
    :param time: List of time points.
    :param pos: List of positions corresponding to time points.
    :param speed: List of speeds corresponding to time points.
    :return: Interpolated position and velocity at target_time.
    """
    index = closest(target_time, time)
    if index == -1:
        return 0, 0
    else:
        t0, t1 = time[index], time[index + 1]
        x0, x1 = pos[index], pos[index + 1]
        v0, v1 = speed[index], speed[index + 1]

        # Normalize the parameter
        interval_length = t1 - t0
        u = (target_time - t0) / interval_length

        # Cubic Hermite basis functions
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2

        position = (
            h00 * x0
            + h10 * interval_length * v0
            + h01 * x1
            + h11 * interval_length * v1
        )

        # Derivatives of basis functions
        dh00 = (6 * u**2 - 6 * u) / interval_length
        dh10 = 3 * u**2 - 4 * u + 1
        dh01 = (-6 * u**2 + 6 * u) / interval_length
        dh11 = 3 * u**2 - 2 * u

        velocity = dh00 * x0 + dh10 * v0 + dh01 * x1 + dh11 * v1

        return position, velocity


lms_values = np.loadtxt("final_assignment/final_LMS.csv", delimiter=";")


def create_hermite_interpolation(time_points, position_values):
    """
    Create an interpolator function using Hermite interpolation.

    :param time_points: List of time points.
    :param position_values: List of positions at each time point.
    :return: A function f(t) that returns the interpolated position.
    """
    n = len(time_points)
    velocity = np.zeros(n)

    # Calculate derivatives using forward differences
    for index in range(n - 1):
        velocity[index] = (position_values[index + 1] - position_values[index]) / (
            time_points[index + 1] - time_points[index]
        )

    # Return a function that evaluates the interpolation at any point
    def interpolator(target_time):
        if target_time < time_points[0] or target_time > time_points[-1]:
            return 0
        pos, _ = hermite_interpolation(
            target_time, time_points, position_values, velocity
        )
        return pos

    return interpolator


# the functions L, M, S
interpolated_L = create_hermite_interpolation(lms_values[:, 0], lms_values[:, 1])
interpolated_M = create_hermite_interpolation(lms_values[:, 0], lms_values[:, 2])
interpolated_S = create_hermite_interpolation(lms_values[:, 0], lms_values[:, 3])


def LMS(wavelength: float) -> np.ndarray:
    """
    Get LMS color response at a given wavelength using interpolated functions.

    :param wavelength: Wavelength in nm.
    :return: Array [L, M, S] values.
    """
    return np.array(
        [
            interpolated_L(wavelength),
            interpolated_M(wavelength),
            interpolated_S(wavelength),
        ]
    )


def Spectrum_to_LMS(spectral_function):
    """
    Convert a spectral function into LMS values via numerical integration.

    :param spectral_function: Function E(λ) defining the spectral power distribution.
    :return: Array [L, M, S] representing LMS color space values.
    """
    # Create array of wavelengths
    wavelengths = np.linspace(390, 830, 500)

    # Evaluate E at each wavelength
    spectrum_values = np.array([spectral_function(xi) for xi in wavelengths])

    # Calculate the L, M, S responses at each wavelength
    lms_responses = np.array([LMS(xi) for xi in wavelengths])

    # Multiply R or G or B with each of L, M, S
    integrands = lms_responses.T * spectrum_values

    # Integrate using trapezoid rule
    dx = (wavelengths[-1] - wavelengths[0]) / (len(wavelengths) - 1)

    # Apply trapezoid rule
    results = np.zeros(3)
    for index in range(3):
        # End points with weight 0.5
        results[index] = 0.5 * dx * (integrands[index][0] + integrands[index][-1])
        # Inner points with weight 1.0
        results[index] += dx * np.sum(integrands[index][1:-1])

    return results


def spectral_red(wavelength):
    """
    Spectral function for red.

    :param wavelength: Wavelength in nm.
    :return: Intensity at that wavelength.
    """
    return np.exp(-((wavelength - 620) ** 2) / 1000) / np.sqrt(2000)


def spectral_green(wavelength):
    """
    Spectral function for green.

    :param wavelength: Wavelength in nm.
    :return: Intensity at that wavelength.
    """
    return np.exp(-((wavelength - 510) ** 2) / 1000) / np.sqrt(2000)


def spectral_blue(wavelength):
    """
    Spectral function for blue.

    :param wavelength: Wavelength in nm.
    :return: Intensity at that wavelength.
    """
    return np.exp(-((wavelength - 460) ** 2) / 1000) / np.sqrt(2000)


# A program to convert (R,G,B) to (L,M,S)
LMS_red = Spectrum_to_LMS(spectral_red)
LMS_green = Spectrum_to_LMS(spectral_green)
LMS_blue = Spectrum_to_LMS(spectral_blue)

rgb_to_lms_matrix = np.column_stack((LMS_red, LMS_green, LMS_blue))
np.linalg.solve(rgb_to_lms_matrix, LMS(580))


def LMS_to_RGB(luminances: np.ndarray):  # 5 points
    """
    Convert LMS color values to gamma-corrected RGB values.

    :param luminances: Array [L, M, S] of luminance values.
    :return: Array [R, G, B]
    """
    # convert_matrix is the matrix that converts RGB to LMS
    # To convert from LMS to RGB, we need the inverse of convert_matrix
    matrix_inverse = np.linalg.inv(rgb_to_lms_matrix)

    # Convert LMS to RGB
    rgb_linear = np.dot(matrix_inverse, luminances)

    # Clip values to ensure they're within a valid range (0-1)
    rgb_linear = np.clip(rgb_linear, 0, 1)

    # Added gamma to get more accurate light
    # Put the gamma for each element
    gammma = 1 / 2.2
    corrected_rgb = np.power(rgb_linear, gammma)

    return (corrected_rgb * 255).astype(int)
