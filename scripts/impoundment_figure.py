import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import xml.etree.ElementTree as ElementTree
from mpmath import *
import math
matplotlib.use('agg')

class impoundment:

    def __init__(self, hydromechanicalParameters, xMin, xMax, appliedTraction):
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
        self.appliedTraction = abs(appliedTraction)
        self.loadingEfficiency = b / (Kv * Se + b**2)
        self.consolidationCoefficient = (k / mu) * Kv / (Se * Kv + b**2)
        self.consolidationTime = self.characteristicLength**2 / self.consolidationCoefficient
        self.initialPressure = self.loadingEfficiency * self.appliedTraction
        self.alpha = b
        self.Kv = Kv

    def computeInitialDisplacement(self, x):
        return (1-self.alpha * self.loadingEfficiency) / self.Kv * self.appliedTraction * (self.characteristicLength - x)

    def computeFinalDisplacement(self, x):
        return (1-self.alpha) / self.Kv * self.appliedTraction * (self.characteristicLength - x)

def getHydromechanicalParametersFromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)

    param1 = tree.find('Constitutive/ElasticIsotropic')
    param2 = tree.find('Constitutive/BiotPorosity')
    param3 = tree.find('Constitutive/CompressibleSinglePhaseFluid')
    param4 = tree.find('Constitutive/ConstantPermeability')

    hydromechanicalParameters = dict.fromkeys([
        "youngModulus", "poissonRation", "biotCoefficient", "fluidViscosity", "fluidCompressibility", "porosity",
        "permeability"
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

    perm = param4.get("permeabilityComponents")
    perm = np.array(perm[1:-1].split(','), float)
    hydromechanicalParameters["permeability"] = perm[0]

    return hydromechanicalParameters


def getAppliedTractionFromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)
    param = tree.find('FieldSpecifications/Traction')
    return float(param.get("scale"))


def getDomainMaxMinXCoordFromXML(xmlFilePath):
    tree = ElementTree.parse(xmlFilePath)
    meshElement = tree.find('Mesh/InternalMesh')
    nodeXCoords = meshElement.get("xCoords")
    nodeXCoords = [float(i) for i in nodeXCoords[1:-1].split(",")]
    xMin = nodeXCoords[0]
    xMax = nodeXCoords[-1]
    return xMin, xMax


def main():
    # File path
    hdf5FilePath = "pressure_history.hdf5"
    xmlFilePath = "./impoundment.xml"

    # Read HDF5
    hf = h5py.File(hdf5FilePath, 'r')
    time = hf.get('pressure Time')
    pressure = hf.get('pressure')
    x = hf.get('pressure elementCenter')
    # read disp HDF5
    hf = h5py.File('disp_history.hdf5', 'r')
    displacement = hf.get('totalDisplacement')
    # Extract info from XML
    hydromechanicalParameters = getHydromechanicalParametersFromXML(xmlFilePath)
    appliedTraction = getAppliedTractionFromXML(xmlFilePath)

    # Get domain min/max coordinate in the x-direction
    xMin, xMax = getDomainMaxMinXCoordFromXML(xmlFilePath)

    # Initialize Terzaghi's analytical solution
    # terzaghiAnalyticalSolution = terzaghi(hydromechanicalParameters, xMin, xMax, appliedTraction)
    impoundmentAnalyticalSolution = impoundment(hydromechanicalParameters, xMin, xMax, appliedTraction)
    # Plot analytical (continuous line) and numerical (markers) pressure solution
    x_analytical = np.linspace(xMin, xMax, 51, endpoint=True)
    pressure_analytical = np.empty(len(x_analytical))

    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=[10,5])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    iplt = -1
    for k in range(0, len(time), 5):
        iplt += 1
        t = time[k, 0]
        ax1.plot(x[k, ::10, 0], pressure[k, ::10], 'o', color=cmap(iplt), label='t = ' + str(t) + ' s')
        ax2.plot(x[k, ::10, 0], displacement[ k, :100:10, 0], 'o',  color=cmap(iplt), label='t = ' + str(t) + ' s')
    x_target = x[0, ::10, 0]
    x_initial = []
    x_end = []
    for x in x_target:
        x_initial.append(impoundmentAnalyticalSolution.computeInitialDisplacement(x))
        x_end.append(impoundmentAnalyticalSolution.computeFinalDisplacement(x))
    ax1.set_xlabel('$x$ [m]')
    ax1.set_ylabel('pressure [Pa]')
    ax1.legend()

    ax2.plot(x_target, x_initial, label='Initial')
    ax2.plot(x_target, x_end, label='Final')
    ax2.set_xlabel('$x$ [m]')
    ax2.set_ylabel('Displacement [m]')
    ax2.legend()
    fig.tight_layout()
    plt.savefig('./impoundment.png')
    plt.close()


if __name__ == "__main__":
    main()
