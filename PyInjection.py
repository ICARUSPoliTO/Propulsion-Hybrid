import CoolProp.CoolProp as cp
import numpy as np
import matplotlib.pyplot as plt

# This script provides a model for double phase fluid injection system, according to
# "Mass Flow Rate and Isolation Characteristics of Injectors for Use with Self-Pressurizing Oxidizers
# in Hybrid Rockets" - Benjamin S. Waxman, Jonah E. Zimmerman, Brian J. Cantwell
# Stanford University, Stanford, CA 94305
# and
# Gregory G. Zilliac
# NASA Ames Research Center, Mo et Field, CA 94035
#

class Injector(object):
    def __init__(self, fluid):
        if fluid in cp.FluidsList():
            self.fluid = fluid
        else:
            print("Fluid not found")
            print(cp.FluidsList())
            exit('Read the list above and try again!')

    def injection_area(self, D, n):
        # D = 'Hole diameter [m]', n = 'Number of holes'
        self.A = 0.25 * n * np.pi * (D ** 2)

    def massflow(self, p1, p2, T, cD):
        # Isothermal fluid in the line (hypothesis)
        # p1 = Tank pressure[Pa], p2 = Chamber pressure[Pa], T = Tank temperature[K]
        h1 = cp.PropsSI('H', 'P',p1,'T',T, self.fluid)
        h2 = cp.PropsSI('H', 'P',p2,'T',T, self.fluid)
        d2 = cp.PropsSI('D', 'P',p2,'T',T, self.fluid)

        dSPI = cp.PropsSI('D', 'T',T, 'Q', 0, self.fluid)

        # Vapor pressure
        pV = cp.PropsSI('P','T',T, 'Q', 1, self.fluid)

        if p1 > p2:
            mdot_SPI = cD * np.sqrt(2 * dSPI * (p1 - p2)) #[kg/s*m^2]

            mdot_HEM = cD * d2 * np.sqrt(2 * abs(h1 - h2)) #[kg/s*m^2]

            k = np.sqrt((p1 - p2) / (pV - p2))

            mdot = (k * mdot_SPI / (k + 1) + mdot_HEM / (k + 1)) #[kg/s*m^2]

            self.mdot = mdot
            # self.mdot_SPI = mdot_SPI * self.A
            # self.mdot_HEM = mdot_HEM * self.A
        else:
            #print('Backflow not possible')
            self.mdot = 0
            self.mdot_SPI = 0
            self.mdot_HEM = 0

if __name__ == '__main__':
    ## Code to verify the injection model and to explain its use
    plt.close('all')

    ox = Injector('NitrousOxide')

    ox.injection_area(0.0127,1)
    pinj= np.arange(70,71,1) #[bar]

    pc = 1 #[bar]
    mdot= np.zeros(np.shape(pinj))
    mdot_SPI= np.zeros(np.shape(pinj))
    mdot_HEM= np.zeros(np.shape(pinj))

    for i in range(np.shape(pinj)[0]):
        ox.massflow( pinj[i]*10**5, pc*10**5, 308, 1)
        mdot[i] = ox.mdot * ox.A

    mfuel = 0.116*(mdot/(0.25*np.pi*(13.4E-3)**2))**0.331

    print(pinj)
    print(mdot)

    #plt.plot(pinj,mdot, label='Dyer')

    #plt.xlabel('Injection pressure [bar]')
    #plt.ylabel('Mass flow [kg/s]')
    #plt.legend()
    #plt.show()

# end of file