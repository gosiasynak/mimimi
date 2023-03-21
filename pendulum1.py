# -*- coding: utf-8 -*-
first_name = 'Malgorzata' # TODO
last_name = 'Synak' # TODO
hand_in = False # TODO: set to True and run again before hand_in. Hand_in also the output* file

import sys
import os
if hand_in:
    filename = os.path.basename(__file__).split('.')[0]
    output = open(f'output_{filename}_{last_name}_{first_name}.txt', 'w')
    sys.stdout = output
    print(f"{filename}\n {last_name}, {first_name}")

import time

import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt


# TODO vervollständigen Sie das Template.

#  Implementieren Sie alle ODE Löser hier. Schreiben Sie
#  erst einen kleinen Teil, z.B. einen Eulerschritt. Testen
#  Sie das Stückchen Code sofort. Möglicherweise anhand einer
#  anderen ODE, z.B. dy/dt = y, denn dafür kennen Sie die exakte
#  Lösung: y(t) = y(0) exp(t).
# 
#  Danach können Sie mehrere Schritte implementieren und mit den
#  anderen Verfahren beginnen. Dabei gilt immernoch: probieren
#  Sie Ihren Code regelmässig aus.


def reference_solver(rhs, T, y0):
    """Berechnet the (fast) exakte Lösung der ODE `dy/dt = rhs(y)`."""

    result = scipy.integrate.solve_ivp(
        rhs, (0.0, T), y0, method="RK45", atol=1e-10, rtol=1e-10
    )
    return result.y[:, -1]



def eE(rhs, y0 , T, N):
    y = np.zeros((N+1,) + y0.shape)
    y[0,:] = y0 
    
    t, h = np.linspace(0, T, N+1, retstep=True)
    
    for k in range(N):
        y[k+1,:] = y[k,:] + h * rhs(t[k], y[k,:])
    return t, y

def iE(rhs, y0 , T, N):
    
    y = np.zeros((N+1,) + y0.shape)
    y[0,:] = y0 # Startwert initialisieren
    
    t, h = np.linspace(0, T, N+1, retstep=True)
    
    for k in range(N):
        F = lambda x: x - y[k,:] - h*rhs(t[k+1], x)
        y[k+1,:] = fsolve(F, y[k,:] + h*rhs(t[k], y[k,:]))

    return t, y

    
def iM(rhs, y0 , T, N):
    return
    
def vV(rhs, y0 , T, N):
    return
    
    
    

    
    
def potential_energy(y):
    """Berechnet die potenzielle Energie von `y`.

    y : 2D-Array. Approximative Lösung der Pendelgleichung.
        Achse 0: Zeitschritte
        Achse 1: Winkel & Geschwindigkeit.
    """

    # TODO vervollständigen Sie das Template.

    return 0.0 # Bitte ersetzen.


def kinetic_energy(y):
    """Berechnet die kinetische Energie von `y`.

    y : 2D-Array. Approximative Lösung der Pendelgleichung.
        Achse 0: Zeitschritte
        Achse 1: Winkel & Geschwindigkeit.
    """

    # TODO vervollständigen Sie das Template.

    return 0.0 # Bitte ersetzen.


def create_plots(y, filename):
    plt.clf()
    plt.plot(y[:, 0], y[:, 1])

    plt.savefig(filename + ".eps")
    plt.savefig(filename + ".png")


def create_energy_plots(t, y, filename):
    E_pot = potential_energy(y)
    E_kin = kinetic_energy(y)
    E_tot = E_pot + E_kin

    plt.clf()
    plt.plot(t, E_pot, label="$E_{pot}$")
    plt.plot(t, E_kin, label="$E_{kin}$")
    plt.plot(t, E_tot, label="$E_{pot}$")
    plt.legend()

    plt.ylabel("Energy")
    plt.xlabel("Time")

    plt.savefig(filename + ".eps")
    plt.savefig(filename + ".png")
    plt.show()



# TODO vervollständigen Sie das Template.

