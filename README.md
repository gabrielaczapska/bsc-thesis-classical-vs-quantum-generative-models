# Comparison of Classical, Hybrid Quantum-Classical, and Quantum Generative Models on the Low-Dimensional Bars and Stripes Dataset

## 1. Project Title and Overview

This repository contains the codebase for a Bachelor's thesis conducted at the Leiden Institute of Advanced Computer Science (LIACS). The project compares three generative models on the low-dimensional 3x3 Bars and Stripes dataset: a classical Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP), a hybrid quantum-classical WGAN-GP, and a Quantum Circuit Born Machine (QCBM).

The goal of this project is a comparison of these models using validity ratio, coverage ratio, and Total Variation (TV) distance. The quantum and hybrid quantum-classical models are evaluated under classical simulation using Pennylane's `default.qubit` state-vector simulator. This allows the behaviour of the models that incorporate quantum circuits to be studied in a controlled environment before deployment on real quantum hardware. These results, therefore, should be interpreted as a preliminary simulation-based assessment rather than as evidence for or against quantum advantage.

### Research Question
"How does the performance of a classical Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) compare to that of a hybrid quantum-classical WGAN-GP and a Quantum Circuit Born Machine (QCBM) in generative modelling, as evaluated using the validity ratio, coverage ratio, and Total Variation distance?"

**Author:** Gabriela Czapska

**Thesis paper:** [thesis.pdf](https://github.com/user-attachments/files/29610002/thesis.pdf)



## 2. Repository Contents

```text
├── README.md
├── LICENSE
├── bars_and_stripes.py
├── evaluation_metrics.py
├── loss_functions_wgan_gp.py
├── plotting.py
├── qcbm.py
├── quantum_wgan_gp.py
├── requirements.txt
├── thesis.pdf
├── training_wgan_gp.py
└── wgan_gp.py
```
- `README.md`: repository documentation.
- `LICENSE`: MIT License terms for the code in this repository.
- `bars_and_stripes.py`: generation and represention of the 3x3 Bars and Stripes dataset.
- `evaluation_metrics.py`: implementation of the evaluation metrics, including validity ratio, coverage ratio, and Total Variation distance.
- `loss_functions_wgan_gp.py`: loss functions (Wasserstein distance and auxiliary loss terms) used for training the WGAN-GP-based models.
- `plotting.py`: visualisation of individual Bars and Stripes patterns, all target patterns, and a target distribution.
- `qcbm.py`: implementation of the Quantum Circuit Born Machine (QCBM).
- `quantum_wgan_gp.py`: implementation of the hybrid quantum-classical WGAN-GP model.
- `requirements.txt`: Python dependencies required to run the project.
- `thesis.pdf`: final BSc thesis document.
- `training_wgan_gp.py`: training framework for WGAN-GP-based models.
- `wgan_gp.py`: implementation of the classical WGAN-GP model.

## 3. Installation

### Prerequisites

This project was developed using Python 3.10.

The required Python packages are:

- NumPy
- PennyLane
- PyTorch
- Matplotlib
- JAX
- Optax

All dependencies are listed in `requirements.txt`.

### Setup

Clone the repository:

```bash
git clone https://github.com/gabrielaczapska/bsc-thesis-classical-vs-quantum-generative-models.git
cd bsc-thesis-classical-vs-quantum-generative-models
```

Create and activate a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 4. How to Run

The models can be trained and evaluated by running the corresponding scripts from the repository root.

To train and evaluate the classical WGAN-GP model:
```bash
python wgan_gp.py
```

To train and evaluate the hybrid quantum-classical WGAN-GP model:
```bash
python quantum_wgan_gp.py
```

To train and evaluate the QCBM model:
```bash
python qcbm.py
```

## 5. Reproducibility, Data & Results

### Dataset
The project uses the 3x3 Bars and Stripes dataset, consisting of 14 valid binary patterns formed by horizontal bars and vertical stripes. 

The dataset is well suited for the classical simulation of quantum and hybrid quantum-classical models because of its low dimensionality, while still representing a non-trivial discrete probability distribution.

<img width="1735" height="537" alt="3110b14980ea3f138271d90acda5f579bc4d183b" src="https://github.com/user-attachments/assets/34cbc759-abc7-45c6-8c29-bec98ac8a7f0" />

_Target probability distribution π(x) over the 14 valid Bars and Stripes patterns. The distribution is uniform, assigning equal probability π(x) = 1/14 ≈ 0.071 to each valid pattern and π(x) = 0 to all remaining 498 states of the 2⁹ = 512 possible 9-pixel states._

<img width="960" height="291" alt="e05a155b4cf047b55eab4025d15d969d16a27571" src="https://github.com/user-attachments/assets/d81daaec-e428-4003-8beb-a9db455d443d" />

_The 14 valid Bars and Stripes patterns on a 3x3 binary grid (black = 0, white = 1). The top row shows the 8 horizontal bar patterns and the bottom row shows the 6 vertical stripe patterns, where each row or column is uniformly black or white._

### Reproducing the Results

The dataset is generated directly in the code when running the corresponding scripts; therefore, no separate raw data file is required.

To reproduce the thesis results, run the corresponding Python model files for the classical WGAN-GP model, the hybrid quantum-classical WGAN-GP model, and the QCBM model, as described in the `How to Run` section. Running these scripts automatically trains and evaluates the models.

Report and average the results over the following 10 independent training seeds: 81, 49, 77, 66, 54, 30, 111, 44, 12, and 21. Compare the models using validity ratio, coverage ratio, and Total Variation (TV) distance.

### Evaluation Metrics
- **Validity Ratio:** The validity ratio measures the fraction of generated samples that belong to the valid Bars and Stripes support. Higher validity indicates that the model generates fewer invalid binary patterns.

- **Coverage Ratio:** The coverage ratio measures the fraction of the 14 valid Bars and Stripes patterns that are generated at least once. Higher coverage indicates better diversity over each batch.

- **Total Variation Distance:** Total Variation, TV, distance measures the difference between the generated probability distribution and the target probability distribution. Lower TV distance indicates that the generated distribution is closer to the target distribution.


### Results
#### Model Comparison on the 3×3 Bars and Stripes Dataset

| Model | Generative Parameters | Validity | Coverage | TV Distance |
|---|---:|---:|---:|---:|
| Classical WGAN-GP | 5,577 | 0.801 ± 0.050 | 0.943 ± 0.066 | 0.252 ± 0.044 |
| Hybrid WGAN-GP | 36 | 0.727 ± 0.076 | 0.893 ± 0.118 | 0.298 ± 0.075 |
| **QCBM** | 162 | **0.927 ± 0.052** | **1.000 ± 0.000** | **0.073 ± 0.052** |

_**Table:** Performance comparison of the considered generative models on the 3×3 Bars and Stripes dataset. The table includes the total number of generative parameters for each model. Values are reported as mean ± standard deviation across 10 independent training seeds. Higher validity and coverage indicate better performance, while lower TV distance indicates closer alignment between the generated and target distributions._

Among the evaluated models, the QCBM achieved the strongest overall performance, with the highest validity ratio, complete support coverage, and the lowest Total Variation distance. This positions it as a promising candidate for future evaluation on real quantum hardware. The classical WGAN-GP consistently outperformed the hybrid quantum-classical WGAN-GP across all evaluation metrics, making the hybrid model the weakest-performing approach in this experiment.

These findings indicate that, for the considered 3x3 Bars and Stripes configuration, the QCBM approximated the target distribution more accurately than either adversarial approach. The results also show that introducing a parametrised quantum circuit into a WGAN-GP generator does not automatically improve generative performance. Instead, the effectiveness of quantum techniques depends on the training framework and the chosen ansatz. 

This emphasises the importance of assessing quantum algorithms on a model-by-model basis, rather than assuming performance improvements from the inclusion of quantum components alone. Therefore the choice of architecture remains crucial.

## 6. Licensing & Acknowledgments
### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

The thesis PDF is included for reference as part of the academic submission.

### Acknowledgements

I would like to sincerely thank my supervisor, Evert van Nieuwenburg, for his support throughout the writing process of this thesis. This work has helped me develop both professionally and personally.
