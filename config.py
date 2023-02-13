import numpy as np
import itertools
import qiskit
import os

#----------------------- Quantum Circuit Settings -----------------------
NUM_QUBITS     = 6
NUM_SHOTS      = 1
NUM_LAYERS     = 6
SHIFT          = np.pi/4

def create_QC_OUTPUTS():
    measurements = list(itertools.product([0, 1], repeat=NUM_QUBITS))
    return [''.join([str(bit) for bit in measurement]) for measurement in measurements]

QC_OUTPUTS     = create_QC_OUTPUTS()
NUM_QC_OUTPUTS = len(QC_OUTPUTS)

SIMULATOR = qiskit.Aer.get_backend('qasm_simulator')

#----------------------- Dataset Settings -----------------------
training_root   = os.path.join( 'datasets', 'EuroSAT', 'training')
validation_root = os.path.join( 'datasets', 'EuroSAT', 'validation')


CLASS_DICT = {
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
EPOCHS          = 100
LEARNING_RATE   = 0.002
MOMENTUM        = 0.5
BATCH_SIZE      = 16
CLASSES         = len(CLASS_DICT)

