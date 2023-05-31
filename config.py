import numpy as np
import itertools
import qiskit
import os

#----------------------- Quantum Circuit Settings -----------------------
NUM_QUBITS      = 4
NUM_SHOTS       = 1 # for timing reasons is set to 1, but in IRL you want this value to be higher https://quantumcomputing.stackexchange.com/questions/9823/what-is-meant-with-shot-in-quantum-computation
NUM_LAYERS      = 5
SHIFT           = np.pi/4

def create_QC_OUTPUTS():
    measurements = list(itertools.product([0, 1], repeat=NUM_QUBITS))
    return [''.join([str(bit) for bit in measurement]) for measurement in measurements]

QC_OUTPUTS      = create_QC_OUTPUTS()
NUM_QC_OUTPUTS  = len(QC_OUTPUTS)

SIMULATOR       = qiskit.Aer.get_backend('qasm_simulator')

#----------------------- Dataset Settings -----------------------
DATASET_ROOT    = os.path.join(os.sep, 'content','EuroSAT')
SPLIT_FACTOR    = 0.2

CLASS_DICT      = {
    "AnnualCrop":           0,
    "Forest":               1,
    "HerbaceousVegetation": 2,
    "Highway":              3,
    "Industrial":           4,
    "Pasture":              5,
    "PermanentCrop":        6,
    "Residential":          7,
    "River":                8,
    "SeaLake":              9
}


#----------------------- Training Settings -----------------------
TRAINING        = True
LOAD_CHECKPOINT = False
EPOCHS          = 20
LEARNING_RATE   = 0.002
MOMENTUM        = 0.5
BATCH_SIZE      = 1
CLASSES         = len(CLASS_DICT)

