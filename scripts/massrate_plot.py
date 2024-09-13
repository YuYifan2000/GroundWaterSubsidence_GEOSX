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
        print('Aquitard Hydralic Diffusivity K_v in paper: m/s', 10**3*9.8*(k / mu))
        print('Aquitard Time,', self.consolidationTime)
        print('Aquitard Hydralic specific storage S_s in paper: 1/m', 10**3*9.8*(Se * Kv + b**2)/Kv)

class aquifer:
    def __init__(self, hydromechanicalParameters, length):
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
        self.width = length
        self.rho = rho
        S = self.alpha / self.loadingEfficiency / self.Kv # specific storage
        self.S = S
        print('aquifer S:',S)
    
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

    hdf5FilePath = "aquifer2_pressure_history.hdf5"
    hf = h5py.File(hdf5FilePath, 'r')
    aquifer2_pressure = hf.get('pressure')
    x2 = hf.get('pressure elementCenter')

    # displacement history hdf5
    hdf5FilePath = "disp_history.hdf5"
    hf = h5py.File(hdf5FilePath, 'r')
    displacement = hf.get('totalDisplacement')
    positionDisp = hf.get('totalDisplacement ReferencePosition')

    # read from xml control file
    xml = './aquifer.xml'
    aquitardParameters, aquiferParameters = getHydromechanicalParametersFromXML(xml)
    xCoord = getDomainMaxMinXCoordFromXML(xml)

    # aquitard & aquifer initialization
    aquitard1 = aquitard(aquitardParameters, xCoord[0], xCoord[1])
    aquifer1 = aquifer(aquiferParameters, 50)
    fig = plt.figure(figsize=[10,6])

    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("tab10")
    timesteps = range(0, len(time), len(time)//10)
    i =0
    for timestep in timesteps:
        x = np.concatenate((x2[timestep, : , 2], x_pressure[timestep,:,2], x1[timestep,:,2]), axis=0)
        p = np.concatenate((aquifer2_pressure[timestep, :], pressure[timestep, :], aquifer1_pressure[timestep, :]), axis=0)
        ax.plot(x, p,color=cmap(i), label=f'{time[timestep, 0] } seconds')
        print(aquifer2_pressure[timestep, 20])
        i += 1
    ax.set_xlabel('z')
    ax.set_ylabel('Pressure (Pa)')
    ax.legend()
    fig.tight_layout()
    plt.savefig('./test.png', dpi=300)
    plt.close()
    

if __name__ == "__main__":
    main()