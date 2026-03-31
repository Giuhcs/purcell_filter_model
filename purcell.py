import numpy as np
from scipy.optimize import curve_fit, fsolve, least_squares
from qibocal.protocols.utils import baseline_als
from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt
from numpy import cos, exp, abs

### auxiliar function to compute spectrum signal from parameters ###
def abs_s_out_in(
    w, # frequency at which to compute the absolute value of the spectrum
    A, # amplitude
    k, # tilt in the spectrum
    w_0, # center of the spectrum
    phi, # phase rotation induced by capacitive coupling to other lines
    k_p, # external coupling rate of the Purcell filter
    w_p, # Purcell filter frequency
    w_r, # resonator frequency, either for ground or excited state
    J, # the transmon resonator coupling rate
):
    return (A + (k*(w-w_0)/w_0 if w_0!=0 else k*w))*abs(cos(phi)-exp(1j*phi)*(k_p*(-2j*(w-w_r)))/(4*J**2+(k_p-2j*(w-w_p))*(-2j*(w-w_r))))

def fit_purcell(
    frequencies,
    data, # signal magnitude for each frequency value
    sigmas, # uncertainties
):
    ##### first we find the peaks to have initial guesses for w_l, w_k, k_l, k_h #####
    # removing baseline to find the peaks
    z = baseline_als(data=data,lamda=1e9,p=0.999)

    # initial guesses for A, k and w_0 from removal
    w_0=frequencies[len(frequencies)//2]
    w_0_guess = w_0
    k_guess = (z[-1]-z[0])/(frequencies[-1]-frequencies[0])
    A_guess = z[len(frequencies)//2]

    # finding the peaks and their widths
    peaks, properties = find_peaks(-(data-z)/abs(min(data-z)),height=0.0, prominence=0.5) # height filters peaks above 0
    rel_height = 0.9
    widths = peak_widths(-(data-z), peaks, rel_height=rel_height)

    # Print results
    print("Peak indices:", peaks)
    print("Peak heights:", -properties["peak_heights"]*abs(min(data-z)))
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
    initial_guess = [A_guess, k_guess, w_0_guess, 0, k_p_guess, w_p_guess, w_r_guess, J_guess] # phi is assumed to be zero here for now

    # auxiliar function to compute the residuals
    def residuals(params, w, y, y_err):
        return (abs_s_out_in(w,*params) - y)/y_err
    
    #fitting original data from these guesses
    result = least_squares(residuals, x0=initial_guess, args=(frequencies, data, sigmas))
    dof = len(data)-len(result.x)
    chi2 = sum(result.fun**2)
    red_chi2 = chi2/dof

    # plotting
    plt.plot(frequencies,abs_s_out_in(frequencies, *result.x), label="Purcell fit")
    plt.errorbar(frequencies,data,yerr=sigmas, label="data",fmt=".", capsize=3, ecolor="gray", alpha=0.4)
    plt.plot(frequencies,z, label="baseline")
    plt.xlabel("w(MHz)")
    plt.ylabel(r"Raw signal ($\mu V$)")
    plt.legend()

    return red_chi2, result.x