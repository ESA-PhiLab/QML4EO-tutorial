# QML4EO-tutorial: Hybrid Quantum Convolutional Neural Network Classifier

This tutorial is meant to be done in google colab


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alessandrosebastianelli/QML4EO-tutorial/blob/main/HQCNN.ipynb)


In any case you can run everything on your hardare, but keep attention to the enviroment (so skip cells that want to install packages) (follow the [requirements file](requirements.txt)). Moreover you need to adjust paths both in [config file](config.py) and at the beginning of the [notebook](notebooks/HQCNN.ipynb).


## Configuration file

Before running the [notebook](notebooks/HQCNN.ipynb) you must check the [config file](config.py). This file is organized in thre sections:

- QCNN settings: here you can modify the quantum cicuit structure by acting on "NUM_QUBITS" and "NUM_LAYERS
- Dataset settings: here you have to set the root path of the training and validation set respectively
- Training settings: here you can modify parameters like learning rate, momentum, etc.

## Main references
- **Sebastianelli, A., Del Rosso, M. P., Ullo, S. L., & Gamba, P. (2023). On Quantum Hyperparameters Selection in Hybrid Classifiers for Earth Observation Data.**
- **Sebastianelli, A., Zaidenberg, D. A., Spiller, D., Le Saux, B., & Ullo, S. L. (2021). On circuit-based hybrid quantum neural networks for remote sensing imagery classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 15, 565-580.**
- **Zaidenberg, D. A., Sebastianelli, A., Spiller, D., Le Saux, B., & Ullo, S. L. (2021, July). Advantages and bottlenecks of quantum machine learning for remote sensing. In 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS (pp. 5680-5683). IEEE.**
- Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
- Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.
- https://qiskit.org/documentation/machine-learning/tutorials/index.html
- https://pennylane.ai/qml/demos_qml.html
