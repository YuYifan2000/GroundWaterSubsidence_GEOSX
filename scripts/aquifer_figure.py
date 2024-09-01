import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import xml.etree.ElementTree as ElementTree
from mpmath import *
import math
matplotlib.use('agg')

class aquitard:
    def __init__(self, hydromechanicalParameters, xMin, xMax):
        E = hydromechanicalParameters["youngModulus"]
        nu = hydromechanicalParameters["poissonRatio"]
        b = hydromechanicalParameters["biotCoefficient"]
        mu = hydromechanicalParameters["fluidViscosity"]
        cf = hydromechanicalParameters["fluidCompressibility"]
        phi = hydromechanicalParameters["porosity"]
        k = hydromechanicalParameters["permeability"]

        K = E / 3.0 / (1.0 - 2.0 * nu)    # bulk modulus
        Kv = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))    # uniaxial bulk modulus
        Se = (b - phi) * (1.0 - b) / K + phi * cf    # constrained specific storage

        self.characteristicLength = xMax - xMin
        self.loadingEfficiency = b / (Kv * Se + b**2)
        self.consolidationCoefficient = (k / mu) * Kv / (Se * Kv + b**2)
        self.consolidationTime = self.characteristicLength**2 / self.consolidationCoefficient
        self.alpha = b
        self.Kv = Kv

        print('Aquitard Biot: ', self.alpha)
        print('Aquitard Hydralic Diffusivity: ', self.consolidationCoefficient)
        print('Aquitard Time,', self.consolidationTime)

class aquifer:
    def __init__(self, hydromechanicalParameters):
        E = hydromechanicalParameters["youngModulus"]
        nu = hydromechanicalParameters["poissonRatio"]
        b = hydromechanicalParameters["biotCoefficient"]
        mu = hydromechanicalParameters["fluidViscosity"]
        cf = hydromechanicalParameters["fluidCompressibility"]
        phi = hydromechanicalParameters["porosity"]
        k = hydromechanicalParameters["permeability"]
        rho = hydromechanicalParameters["fluidDensity"]

        K = E / 3.0 / (1.0 - 2.0 * nu)    # bulk modulus
        Kv = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))    # uniaxial bulk modulus
        Se = (b - phi) * (1.0 - b) / K + phi * cf    # constrained specific storage

        self.loadingEfficiency = b / (Kv * Se + b**2)
        self.consolidationCoefficient = (k / mu) * Kv / (Se * Kv + b**2)
        self.alpha = b
        self.Kv = Kv
        self.width = 50
        self.rho = rho
        S = self.alpha / self.loadingEfficiency / self.Kv # specific storage
        self.S = S
    
    def computeNormalStress(self, dh):
        g = 9.8
        return self.rho * g * self.S * self.width * self.rho * g * dh

def getHydromechanicalParametersFromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)
    root = tree.getroot()
    param1 = tree.find('Constitutive/ElasticIsotropic')
    param2 = tree.find('Constitutive/BiotPorosity')
    param3 = tree.find('Constitutive/CompressibleSinglePhaseFluid')
    param4 = tree.find('Constitutive/ConstantPermeability')
    aquitardHydromechanicalParameters = dict.fromkeys([
        "youngModulus", "poissonRation", "biotCoefficient", "fluidViscosity", "fluidCompressibility", "porosity",
        "permeability", "fluidDensity"
    ])
    aquiferHydromechanicalParameters = dict.fromkeys([
        "youngModulus", "poissonRation", "biotCoefficient", "fluidViscosity", "fluidCompressibility", "porosity",
        "permeability", "fluidDensity"
    ])
    # solid
    for field_spec in root.findall('.//ElasticIsotropic'):
        name = field_spec.get('name')
        if name == 'skeletonAquitard':
            aquitardHydromechanicalParameters["youngModulus"] = float(field_spec.get('defaultYoungModulus'))
            aquitardHydromechanicalParameters["poissonRatio"] = float(field_spec.get("defaultPoissonRatio"))
        if name == 'skeletonAquifer':
            aquiferHydromechanicalParameters["youngModulus"] = float(field_spec.get('defaultYoungModulus'))
            aquiferHydromechanicalParameters["poissonRatio"] = float(field_spec.get("defaultPoissonRatio"))
    # biot
    for field_spec in root.findall('.//BiotPorosity'):
        name = field_spec.get('name')
        if name == 'aquitardPorosity':
            aquitardHydromechanicalParameters["porosity"] = float(field_spec.get('defaultReferencePorosity'))
            Kg = float(field_spec.get("defaultGrainBulkModulus"))
            E = aquitardHydromechanicalParameters["youngModulus"]
            nu = aquitardHydromechanicalParameters["poissonRatio"]
            K = E / 3.0 / (1.0 - 2.0 * nu)
            aquitardHydromechanicalParameters["biotCoefficient"] = 1.0 - K / Kg
        if name == 'aquiferPorosity':
            aquiferHydromechanicalParameters["porosity"] = float(field_spec.get('defaultReferencePorosity'))
            Kg = float(field_spec.get("defaultGrainBulkModulus"))
            E = aquiferHydromechanicalParameters["youngModulus"]
            nu = aquiferHydromechanicalParameters["poissonRatio"]
            K = E / 3.0 / (1.0 - 2.0 * nu)
            aquiferHydromechanicalParameters["biotCoefficient"] = 1.0 - K / Kg

    aquitardHydromechanicalParameters["fluidViscosity"] = float(param3.get("defaultViscosity"))
    aquitardHydromechanicalParameters["fluidCompressibility"] = float(param3.get("compressibility"))
    aquitardHydromechanicalParameters["fluidDensity"] = float(param3.get("defaultDensity"))

    aquiferHydromechanicalParameters["fluidViscosity"] = float(param3.get("defaultViscosity"))
    aquiferHydromechanicalParameters["fluidCompressibility"] = float(param3.get("compressibility"))
    aquiferHydromechanicalParameters["fluidDensity"] = float(param3.get("defaultDensity"))

    # permeability
    for field_spec in root.findall('.//ConstantPermeability'):
        name = field_spec.get('name')
        if name == 'aquitardPerm':
            perm = field_spec.get('permeabilityComponents')
            perm = np.array(perm[1:-1].split(','), float)
            aquitardHydromechanicalParameters["permeability"] = perm[0]
        if name == 'aquiferPerm':
            perm = field_spec.get('permeabilityComponents')
            perm = np.array(perm[1:-1].split(','), float)
            aquiferHydromechanicalParameters["permeability"] = perm[0]

    return aquitardHydromechanicalParameters, aquiferHydromechanicalParameters

def getDomainMaxMinXCoordFromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)
    meshElement = tree.find('Mesh/InternalMesh')
    nodeXCoords = meshElement.get("xCoords")
    nodeXCoords = [float(i) for i in nodeXCoords[1:-1].split(",")]

    return nodeXCoords

def main():
    # aquitard pressure history hdf5
    hdf5FilePath = "pressure_history.hdf5"
    hf = h5py.File(hdf5FilePath, 'r')
    time = hf.get('pressure Time')
    pressure = hf.get('pressure')
    x_pressure = hf.get('pressure elementCenter')

    # aquifer pressure history hdf5
    hdf5FilePath = "aquifer_pressure_history.hdf5"
    hf = h5py.File(hdf5FilePath, 'r')
    aquifer1_pressure = hf.get('pressure')
    x1 = hf.get('pressure elementCenter')

    # displacement history hdf5
    hdf5FilePath = "disp_history.hdf5"
    hf = h5py.File(hdf5FilePath, 'r')
    displacement = hf.get('totalDisplacement')

    # read from xml control file
    xml = './aquifer.xml'
    aquitardParameters, aquiferParameters = getHydromechanicalParametersFromXML(xml)
    xCoord = getDomainMaxMinXCoordFromXML(xml)

    # aquitard & aquifer initialization
    aquitard1 = aquitard(aquitardParameters, xCoord[1], xCoord[2])
    aquifer1 = aquifer(aquiferParameters)
    fig = plt.figure(figsize=[10,5])
    ax = fig.add_subplot(131)
    ax.plot(time[:,0] / 86400, displacement[:,0,0],  label='GEOSX')
    ax.set_xlabel('Time (day)')
    ax.set_ylabel('Displacement (m) at top of aquitard')
    ax.legend()

    ax = fig.add_subplot(132)
    ax.plot(x_pressure[-2, :, 0], pressure[-2, :], label='GEOSX')
    ax.set_xlabel('x')
    ax.set_ylabel('Pressure (Pa) at Aquitard')
    ax.legend()

    ax = fig.add_subplot(133)
    ax.plot(x1[-2, :, 0], aquifer1_pressure[-2, :], label='GEOSX')
    ax.set_xlabel('x')
    ax.set_ylabel('Pressure (Pa) at Aquifer')
    ax.legend()
    fig.tight_layout()
    plt.savefig('./aquifer_aquitard.png', dpi=300)
    plt.close()
    u_0_t= np.zeros([len(time),1])
    
#    for i in range(0, len(time)):
#        t = time[i,0]
#        int_p = sum(pressure[i, :]) * (length / pressure.shape[1])
#        u_0_t[i] = - (-sigma * length + aquitard1.alpha * int_p) / aquitard1.Kv

#    x_t_0 = aquifer1.rho * 9.8 * aquifer1.S * aquifer1.width*(1 - aquitard1.alpha * aquitard1.loadingEfficiency)*length/aquitard1.Kv*p1
#    x_t_infty = -(p1+p2)/2*aquitard1.alpha*length/aquitard1.Kv + p1*aquifer1.rho * 9.8 * aquifer1.S * aquifer1.width*length/aquitard1.Kv
#    ax.hlines(x_t_0, xmin=0, xmax=300, linestyles='--', label='Undrained Response', colors='k')
#    ax.hlines(x_t_infty, xmin=0, xmax=300, linestyles='--', label='Steady State', colors='k')

if __name__ == "__main__":
    main()