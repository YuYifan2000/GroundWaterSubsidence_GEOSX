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
        nu = hydromechanicalParameters["poissonRation"]
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

class aquifer:
    def __init__(self, hydromechanicalParameters):
        E = hydromechanicalParameters["youngModulus"]
        nu = hydromechanicalParameters["poissonRation"]
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
    
    def computeNormalStress(self, dh):
        S = self.alpha / self.loadingEfficiency / self.Kv
        g = 9.8
        return self.rho * g * S * self.width

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
    hydromechanicalParameters["poissonRation"] = float(param1.get("defaultPoissonRatio"))

    E = hydromechanicalParameters["youngModulus"]
    nu = hydromechanicalParameters["poissonRation"]
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
    for i in range(0, len(time)):
        t = time[i,0]
        int_p = sum(pressure[i, :]) * (length / pressure.shape[1])
        u_0_t[i] = - (sigma * length + aquitard1.alpha * int_p) / aquitard1.Kv
    fig = plt.figure(figsize=[10,5])
    ax = fig.add_subplot(121)
    ax.plot(time[:,0], u_0_t, label='Analytical')
    ax.plot(time[:,0], displacement[:,0,0],  label='GEOSX', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Displacement (m) at x=0')
    ax.legend()

    ax = fig.add_subplot(122)
    ax.plot(x_pressure[-2, :, 0], pressure[-2, :], label='GEOSX')
    ax.scatter([xMin, xMax], [p1, p2], label='Analytical', c='k')
    ax.set_xlabel('x')
    ax.set_ylabel('Pressure (Pa)')
    ax.legend()
    plt.savefig('./pumping.png')
    plt.close()
if __name__ == "__main__":
    main()
    # xml = './pumping.xml'
    # hydromechanicalParameters = getHydromechanicalParametersFromXML(xml)
    # xMin, xMax = getDomainMaxMinXCoordFromXML(xml)
    # aquifer1 = aquifer(hydromechanicalParameters)
    # print(aquifer1.computeNormalStress(1))