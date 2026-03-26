import numpy as np
from scipy.optimize import curve_fit, fsolve, least_squares
from qibocal.protocols.utils import baseline_als
from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt

def fit_purcell(
    frequencies,
    data, # signal magnitude for each frequency value
    sigmas, # uncertainties
):
    ##### first step we find an approximate linear form to the baseline for an initial guess to A, k and w_0 #####
    def linear_baseline(w, A, k, w_0):
        return k * (w - w_0) + A

    baseline_params, covariance = curve_fit(
        linear_baseline,
        frequencies,
        data,
        sigma=sigmas,
        absolute_sigma=True)

    A_guess, k_guess, w_0_guess = baseline_params # this will be used as initial guesses

    ##### secondly, we find the peaks to have initial guesses for w_l, w_k, k_l, k_h #####
    # removing baseline to find the peaks
    z = baseline_als(data=data,lamda=1e9,p=0.99)

    # finding the peaks and their widths
    peaks, properties = find_peaks(-(data-z),height=0.0) # height filters peaks above 0
    rel_height = 0.9
    widths = peak_widths(-(data-z), peaks, rel_height=rel_height)

    # Print results
    print("Peak indices:", peaks)
    print("Peak heights:", -properties["peak_heights"])
    print(f"Peak widths at {rel_height*100 :.0f}% height:", widths[0])

    # determining the guesses
    w_l_guess, w_h_guess = frequencies[peaks]
    k_l_guess, k_h_guess = widths[0]

    ##### using (2) in https://arxiv.org/pdf/2307.07765 to initial guesses for w_r, w_p, k_p and J #####
    # auxiliar function to solve system of equations
    def equations(vars):
        w_r, w_p, k_p, J = vars

        expr = np.sqrt((w_r - w_p + 1j * k_p * 0.5)**2 + 4 * (J**2))

        eq1 = 0.5*(w_r + w_p) + 0.5*np.real(expr) - w_h_guess
        eq2 = 0.5*(w_r + w_p) - 0.5*np.real(expr) - w_l_guess
        eq3 = 0.5*k_p - np.imag(expr) - k_h_guess
        eq4 = 0.5*k_p + np.imag(expr) - k_l_guess

        return [eq1, eq2, eq3, eq4]

    # solving and getting the intitial guesses for w_r, w_p, k_p and J
    solution = fsolve(equations, [1, 1, 1, 1])
    w_r_guess, w_p_guess, k_p_guess, J_guess = solution

    ##### wrapping up all the initial guesses and fitting the model to the original data #####
    # initial guess
    initial_guess = [A_guess, k_guess, w_0_guess, 0, k_p_guess, w_p_guess, w_r_guess, J_guess] # phi is assumed to be zero here

    # auxiliar function to compute the residuals
    def residuals(params, w, y, y_err):
        A,k,w_0,phi,k_p,w_p,w_res,J = params
        return ((A + k*(w-w_0))*abs(np.cos(phi)-np.exp(1j*phi)*(k_p*(-2j*(w-w_res)))/(4*J**2+(k_p-2j*(w-w_p))*(-2j*(w-w_res)))) - y)/y_err
    
    #fitting original data from these guesses
    result = least_squares(residuals, x0=initial_guess, args=(frequencies, data, sigmas))
    dof = len(data)-len(result.x)
    chi2 = sum(result.fun**2)
    red_chi2 = chi2/dof

    ##### defining the fitting function for plotting ends #####
    A,k,w_0,phi,k_p,w_p,w_r,J = result.x
    fit = lambda w: (A + k*(w-w_0))*abs(np.cos(phi)-np.exp(1j*phi)*(k_p*(-2j*(w-w_r)))/(4*J**2+(k_p-2j*(w-w_p))*(-2j*(w-w_r))))

    plt.plot(frequencies,fit(frequencies))
    plt.errorbar(frequencies,data,yerr=sigmas)

    return fit, red_chi2, result.x