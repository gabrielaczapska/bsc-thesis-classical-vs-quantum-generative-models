# Comparison of Classical, Hybrid Quantum-Classical, and Quantum Generative Models on the Low-Dimensional Bars and Stripes Dataset 
This repository contains the codebase for a Bachelor's Thesis conducted at the Leiden Institute of Advanced Computer Science (LIACS). This project compares three generative modelling approaches on the low-dimensional 3x3 Bars and Stripes dataset:

- a classical Wasserstein Generative Adversarial Network with Gradient Penalty, WGAN-GP
- a hybrid quantum-classical WGAN-GP
- a Quantum Circuit Born Machine, QCBM.

The quantum and hybrid quantum-classical models are evaluated under classical simulation that uses Pennylane's default.qubit state-vector simulator. This allows to study the behaviour of the models that incorporate quantum circuits in a controlled environment before the deployment on real quantum hardware. These results, therefore, should be interpreted as a preliminary simulation-based assessment rather than as evidence for or against quantum advantage.

Author: Gabriela Czapska

Thesis Paper: [Classical_vs_Quantum_Comparison.pdf](https://github.com/user-attachments/files/29594744/Classical_vs_Quantum_Comparison.pdf)

### Abstract:
This thesis investigates classical, hybrid quantum-classical, and quantum approaches to generative modelling using the low-dimensional \(3 \times 3\) Bars and Stripes dataset. Three models are compared: a classical Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP), a hybrid quantum-classical WGAN-GP, and a Quantum Circuit Born Machine (QCBM). The models are evaluated using \textit{support validity}, \textit{support coverage}, and \textit{Total Variation (TV) distance} in order to assess the structural validity and diversity of generated samples within each batch, as well as the similarity between the learned and target probability distributions.

The WGAN-GP-based models are trained using adversarial optimisation, whereas the QCBM optimises the probability distribution encoded in the quantum circuit by minimising the kernel-based Maximum Mean Discrepancy (MMD) loss. The quantum models are implemented with PennyLane using the \texttt{default.qubit} state-vector simulator. Therefore, the results should be interpreted as a preliminary assessment of quantum approaches under classical simulation against a classical baseline, rather than as evidence for or against quantum advantage.

The QCBM achieved the strongest overall performance, with the highest validity ratio, complete support coverage, and the lowest Total Variation distance. The classical WGAN-GP consistently outperformed the hybrid quantum-classical WGAN-GP, suggesting that the inclusion of a parametrised quantum circuit (PQC) within an adversarial generator did not provide a measurable improvement for this task. Overall, the results show that quantum and hybrid quantum-classical generative models can learn non-trivial discrete probability distributions but that their effectiveness depends strongly on the selected quantum model, ansatz, and training objective. Future research should investigate whether these findings persist for larger datasets, more complex distributions, and implementations on real quantum hardware.

Supervisor: Evert van Nieuwenburg

### Research Question
"How does the performance of a classical Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) compare to that of a hybrid quantum-classical WGAN-GP and a Quantum Circuit Born Machine (QCBM) in generative modelling, as evaluated using the validity ratio, coverage ratio, and Total Variation distance?"

### Dataset
The project uses the Bars and Stripes dataset, consisting of 14 valid binary 3×3 patterns formed by horizontal bars and vertical stripes. 

The dataset is well suited for the classical simulation of quantum and hybrid quantum-classical models because of its low dimensionality, while still representing a non-trivial discrete probability distribution.

<img width="1735" height="537" alt="3110b14980ea3f138271d90acda5f579bc4d183b" src="https://github.com/user-attachments/assets/34cbc759-abc7-45c6-8c29-bec98ac8a7f0" />

_Target probability distribution $\pi(x)$ over the 14 valid Bars and Stripes patterns. The distribution is uniform, assigning equal probability $\pi(x) = \frac{1}{14} \approx 0.071$ to each valid pattern and $\pi(x) = 0$ to all remaining 498 states of the $2^9 = 512$ possible 9-pixel states._

<img width="960" height="291" alt="e05a155b4cf047b55eab4025d15d969d16a27571" src="https://github.com/user-attachments/assets/d81daaec-e428-4003-8beb-a9db455d443d" />

_The 14 valid Bars and Stripes patterns on a $3 \times 3$ binary grid (black = 0, white = 1). The top row shows the 8 horizontal bar patterns and the bottom row shows the 6 vertical stripe patterns, where each row or column is uniformly black or white._

## Evaluation Metrics
- Validity Ratio
The validity ratio measures the fraction of generated samples that belong to the valid Bars and Stripes support.

Higher validity indicates that the model generates fewer invalid binary patterns.

- Coverage Ratio
The coverage ratio measures the fraction of the 14 valid Bars and Stripes patterns that are generated at least once.

Higher coverage indicates better diversity over each batch.

- Total Variation Distance
Total Variation, TV, distance measures the difference between the generated probability distribution and the target probability distribution.

Lower TV distance indicates that the generated distribution is closer to the target distribution.

## Repository Structure

## Installation

## Usage

### Reproducibility

All reported results are averaged over 10 independent training seeds: 81, 49, 77, 66, 54, 30, 111, 44, 12, and 21.

### Results
#### Model Comparison on the 3×3 Bars and Stripes Dataset

| Model | Generative Parameters | Validity | Coverage | TV Distance |
|---|---:|---:|---:|---:|
| Classical WGAN-GP | 5,577 | 0.801 ± 0.050 | 0.943 ± 0.066 | 0.252 ± 0.044 |
| Hybrid WGAN-GP | 36 | 0.727 ± 0.076 | 0.893 ± 0.118 | 0.298 ± 0.075 |
| **QCBM** | 162 | **0.927 ± 0.052** | **1.000 ± 0.000** | **0.073 ± 0.052** |

**Table:** Performance comparison of the considered generative models on the 3×3 Bars and Stripes dataset. The table includes the total number of generative parameters for each model. Values are reported as mean ± standard deviation across 10 independent training seeds. Higher validity and coverage indicate better performance, while lower TV distance indicates closer alignment between the generated and target distributions.

Among the evaluated models, the QCBM achieved the strongest overall performance, with the highest validity ratio, complete support coverage, and the lowest Total Variation distance, positioning it as a promising candidate for future evaluation on real quantum hardware.. The classical WGAN-GP consistently outperformed the hybrid quantum-classical WGAN-GP across all evaluation metrics, making the hybrid model the weakest-performing approach in this experiment.

These findings indicate that, for the considered $3 \times 3$ Bars and Stripes configuration, the QCBM approximated the target distribution more accurately than either adversarial approach. The results also show that introducing a parametrised quantum circuit into a WGAN-GP generator does not automatically improve generative performance. Instead, the effectiveness of quantum techniques depends on the training framework and the chosen ansatz. 

This emphasises the importance of assessing quantum algorithms on a model-by-model basis, rather than assuming performance improvements from the inclusion of quantum components alone, therefore the choice of architecture remains crucial.

### Limitations
This repository reports results on the low-dimensional 3×3 Bars and Stripes dataset. While this dataset is useful for controlled experiments with quantum and hybrid quantum-classical generative models, the results may not generalise to larger or more complex datasets.

All quantum and hybrid quantum-classical experiments are performed using classical simulation rather than real quantum hardware. Therefore, the results do not account for hardware noise, decoherence, limited qubit connectivity, or other constraints present in current NISQ devices.

The quantum-based models also use different circuit architectures and training objectives, so performance differences should be interpreted as comparisons between the implemented models rather than as general conclusions about classical, hybrid, or quantum generative modelling as a whole.

### Future Work
Several directions for future research can be identified.

First, the investigated models should be evaluated on larger datasets and more complex probability distributions to assess their scalability beyond low-dimensional settings.

Second, alternative quantum ansätze and training objectives could be explored to improve the performance of hybrid quantum-classical adversarial models.

Third, experiments on Noisy Intermediate-Scale Quantum, NISQ, hardware would provide valuable insight into the practical feasibility of the investigated approaches under realistic hardware constraints, before fault-tolerant quantum computers become available.

Finally, future work could examine whether the performance advantages observed for QCBMs under classical simulation persist as problem dimensionality, dataset complexity, and circuit size increase.
