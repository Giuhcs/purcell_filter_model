import numpy as np
from scipy.optimize import curve_fit, fsolve, least_squares
from qibocal.protocols.utils import baseline_als
from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt
from numpy import cos, exp, abs

### auxiliar function to compute signal from parameters ###
def s_out_in(
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
    return (A + (k*(w-w_0)/w_0 if w_0!=0 else k*w))*(cos(phi)-exp(1j*phi)*(k_p*(-2j*(w-w_r)))/(4*J**2+(k_p-2j*(w-w_p))*(-2j*(w-w_r))))

def fit_purcell(
    frequencies,
    data, # signal for each frequency value
    #sigmas, # uncertainties
):
    phases = np.angle(data).copy() # used below only for plotting ends
    data = abs(data)

    ##### first we find the peaks to have initial guesses for w_l, w_k, k_l, k_h #####
    # removing baseline to find the peaks
    z = baseline_als(data=data,lamda=1e9,p=0.999)

    # finding the peaks and their widths
    peaks, properties = find_peaks(-(data-z)/abs(min(data-z)),height=0.0, prominence=0.5) # height filters peaks above 0
    rel_height = 0.9 # no particular reason for this choice
    widths = peak_widths(-(data-z), peaks, rel_height=rel_height)

    # Print results
    print("Peak indices:", peaks)
    print("Peak heights:", -properties["peak_heights"]*abs(min(data-z)))
    print(f"Peak widths at {rel_height*100 :.0f}% height:", widths[0])

    # determining the guesses
    w_l_guess, w_h_guess = frequencies[peaks]
    k_l_guess, k_h_guess = widths[0]

    ##### using (2) in https://arxiv.org/pdf/2307.07765 to get initial guesses for w_r, w_p, k_p and J #####
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
    phi_guess = phases[np.argmax(frequencies - w_r_guess)] # phi_guess is taken as the phase of the signal point corresponding to the furthest frequency to w_r
    initial_guess = [phi_guess, k_p_guess, w_p_guess, w_r_guess, J_guess]

    # model to fit
    def model(w,phi,k_p,w_p,w_r,J): # relevant params include the ones not in the linear baseline only
        return abs(s_out_in(w,1,0,0,phi,k_p,w_p,w_r,J)) # model assume data to be fit is normalized by baseline

    # fitting data normalized by baseline from these guesses using curve_fit
    popt, pcov = curve_fit(model,frequencies,data/z,p0=initial_guess)

    # plotting 2
    # Create figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    params = [1,0,0,*popt] # 1 because we assume fit from data normalized by baseline

    # Left subplot: fit, baseline and amplitude data
    axes[0].plot(frequencies,abs(s_out_in(frequencies, *params))*z, label="Purcell fit")
    axes[0].scatter(frequencies,abs(data),label="data",alpha=0.3, marker=".", c="orange")
    axes[0].plot(frequencies,z, label="baseline from ALS", c="green")
    axes[0].set_ylabel(r"Amplitude($\mu V$)")
    axes[0].set_xlabel("w(MHz)")
    axes[0].legend()

    # Right subplot: phase fit and data
    axes[1].plot(frequencies,np.angle(s_out_in(frequencies, *params)), label="Purcell fit")
    axes[1].scatter(frequencies,phases,label="data",alpha=0.3, marker=".", c="orange")
    axes[1].set_ylabel("Phase(rad)")
    axes[1].set_xlabel("w(MHz)")
    axes[1].legend()

    return pcov, popt