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
        self.width = 25
        self.rho = rho
        S = self.alpha / self.loadingEfficiency / self.Kv # specific storage
        self.S = S
    
    def computeNormalStress(self, dh):
        g = 9.8
        return self.rho * g * self.S * self.width * self.rho * g * dh

def getHydromechanicalParametersFromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)

    param1 = tree.find('Constitutive/ElasticIsotropic')
    param2 = tree.find('Constitutive/BiotPorosity')
    param3 = tree.find('Constitutive/CompressibleSinglePhaseFluid')
    param4 = tree.find('Constitutive/ConstantPermeability')

    hydromechanicalParameters = dict.fromkeys([
        "youngModulus", "poissonRation", "biotCoefficient", "fluidViscosity", "fluidCompressibility", "porosity",
        "permeability", "fluidDensity"
    ])

    hydromechanicalParameters["youngModulus"] = float(param1.get("defaultYoungModulus"))
    hydromechanicalParameters["poissonRatio"] = float(param1.get("defaultPoissonRatio"))

    E = hydromechanicalParameters["youngModulus"]
    nu = hydromechanicalParameters["poissonRatio"]
    K = E / 3.0 / (1.0 - 2.0 * nu)
    Kg = float(param2.get("defaultGrainBulkModulus"))

    hydromechanicalParameters["biotCoefficient"] = 1.0 - K / Kg
    hydromechanicalParameters["porosity"] = float(param2.get("defaultReferencePorosity"))
    hydromechanicalParameters["fluidViscosity"] = float(param3.get("defaultViscosity"))
    hydromechanicalParameters["fluidCompressibility"] = float(param3.get("compressibility"))
    hydromechanicalParameters["fluidDensity"] = float(param3.get("defaultDensity"))

    perm = param4.get("permeabilityComponents")
    perm = np.array(perm[1:-1].split(','), float)
    hydromechanicalParameters["permeability"] = perm[0]

    return hydromechanicalParameters

def aquiferHydromechanicalParametersFromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)
    param2 = tree.find('Constitutive/BiotPorosity')
    param3 = tree.find('Constitutive/CompressibleSinglePhaseFluid')

    hydromechanicalParameters = dict.fromkeys([
        "youngModulus", "poissonRation", "biotCoefficient", "fluidViscosity", "fluidCompressibility", "porosity",
        "permeability", "fluidDensity"
    ])

    hydromechanicalParameters["youngModulus"] = 1e10
    hydromechanicalParameters["poissonRatio"] = 0.25

    E = hydromechanicalParameters["youngModulus"]
    nu = hydromechanicalParameters["poissonRatio"]
    K = E / 3.0 / (1.0 - 2.0 * nu)
    Kg = 5e10

    hydromechanicalParameters["biotCoefficient"] = 1.0 - K / Kg
    hydromechanicalParameters["porosity"] = 0.2
    hydromechanicalParameters["fluidViscosity"] = float(param3.get("defaultViscosity"))
    hydromechanicalParameters["fluidCompressibility"] = float(param3.get("compressibility"))
    hydromechanicalParameters["fluidDensity"] = float(param3.get("defaultDensity"))
    hydromechanicalParameters["permeability"] = 1e-12

    return hydromechanicalParameters


def getDomainMaxMinXCoordFromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)
    meshElement = tree.find('Mesh/InternalMesh')
    nodeXCoords = meshElement.get("xCoords")
    nodeXCoords = [float(i) for i in nodeXCoords[1:-1].split(",")]
    xMin = nodeXCoords[0]
    xMax = nodeXCoords[-1]
    return xMin, xMax
def getSigmaFromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)
    param = tree.find('FieldSpecifications/Traction')
    return float(param.get("scale"))

def getP1P2FromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)
    root = tree.getroot()
    # Define the names you want to search for
    target_names = ["xnegboundaryPressure", "xposboundaryPressure"]
    # Iterate through each FieldSpecification element to find the ones with matching names
    for field_spec in root.findall('.//FieldSpecification'):
        name = field_spec.get('name')
        if name == target_names[0]:
            scale = field_spec.get('scale')
            p1 = float(scale)
        if name == target_names[1]:
            scale = field_spec.get('scale')
            p2 = float(scale)
    return p1,p2
def main():
    # pressure history hdf5
    hdf5FilePath = "pressure_history.hdf5"
    hf = h5py.File(hdf5FilePath, 'r')
    time = hf.get('pressure Time')
    pressure = hf.get('pressure')
    x_pressure = hf.get('pressure elementCenter')

    # displacement history hdf5
    hdf5FilePath = "disp_history.hdf5"
    hf = h5py.File(hdf5FilePath, 'r')
    displacement = hf.get('totalDisplacement')

    # read from xml control file
    xml = './pumping.xml'
    hydromechanicalParameters = getHydromechanicalParametersFromXML(xml)
    xMin, xMax = getDomainMaxMinXCoordFromXML(xml)
    sigma = getSigmaFromXML(xml)
    p1, p2 = getP1P2FromXML(xml)
    length = xMax - xMin
    # aquitard initialization
    aquitard1 = aquitard(hydromechanicalParameters, xMin, xMax)
    u_0_t= np.zeros([len(time),1])
    # displacement comparison
    print('Sigma: ', sigma)
    for i in range(0, len(time)):
        t = time[i,0]
        int_p = sum(pressure[i, :]) * (length / pressure.shape[1])
        u_0_t[i] = - (-sigma * length + aquitard1.alpha * int_p) / aquitard1.Kv
    fig = plt.figure(figsize=[10,5])
    ax = fig.add_subplot(121)
    # ax.plot(time[:,0], u_0_t, label='Analytical')
    # aquifer response
    xml = './pumping.xml'
    hydromechanicalParameters = getHydromechanicalParametersFromXML(xml)
    xMin, xMax = getDomainMaxMinXCoordFromXML(xml)
    aquifer1 = aquifer(hydromechanicalParameters)
    x_t_0 = aquifer1.rho * 9.8 * aquifer1.S * aquifer1.width*(1 - aquitard1.alpha * aquitard1.loadingEfficiency)*length/aquitard1.Kv*p1
    x_t_infty = -(p1+p2)/2*aquitard1.alpha*length/aquitard1.Kv + p1*aquifer1.rho * 9.8 * aquifer1.S * aquifer1.width*length/aquitard1.Kv
    ax.hlines(x_t_0, xmin=0, xmax=300, linestyles='--', label='Undrained Response', colors='k')
    ax.hlines(x_t_infty, xmin=0, xmax=300, linestyles='--', label='Steady State', colors='k')

    ax.plot(time[:,0] / 86400, displacement[:,0,0],  label='GEOSX')
    ax.set_xlabel('Time (day)')
    ax.set_ylabel('Displacement (m) at x=0')
    ax.legend()

    ax = fig.add_subplot(122)
    ax.plot(x_pressure[-2, :, 0], pressure[-2, :], label='GEOSX')
    ax.scatter([xMin, xMax], [p1, p2], label='Analytical', c='k')
    ax.set_xlabel('x')
    ax.set_ylabel('Pressure (Pa)')
    ax.legend()
    plt.savefig('./pumping.png', dpi=300)
    plt.close()
if __name__ == "__main__":
    main()
    # xml = './pumping.xml'
    # hydromechanicalParameters = getHydromechanicalParametersFromXML(xml)
    # xMin, xMax = getDomainMaxMinXCoordFromXML(xml)
    # aquifer1 = aquifer(hydromechanicalParameters)
    # print('aquifer normal stress: ')
    # print(aquifer1.computeNormalStress(1))
    xml = './pumping.xml'
    hydromechanicalParameters = getHydromechanicalParametersFromXML(xml)
    xMin, xMax = getDomainMaxMinXCoordFromXML(xml)
    # aquitard initialization
    aquitard1 = aquitard(hydromechanicalParameters, xMin, xMax)
    aquifer1 = aquifer(aquiferHydromechanicalParametersFromXML(xml))
    print('aquifer normal stress: ')
    print(aquifer1.computeNormalStress( 5))