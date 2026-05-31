# Comparison of Classical, Hybrid Quantum-Classical, and Quantum Generative Models on the Low-Dimensional Bars and Stripes Dataset 
This project contains the codebase for a Bachelor's Thesis conducted at the Leiden Institute of Advanced Computer Science (LIACS). The study uses a classical Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) as a baseline for comparison against a Quantum Circuit Born Machine (QCBM) and a hybrid quantum-classical WGAN-GP model.
The quantum and hybrid quantum-classical models are evaluated in simulation, providing a controlled environment for studying the behaviour of models based on the principles of quantum mechanics prior to deployment on real quantum hardware. Performance is compared using metrics such as Total Variation Distance, Support Coverage, Support Validity, and the empirical distribution over valid patterns.
The dataset used for training and evaluation is bars_and_stripes.py, which consists of 14 valid 3×3 binary patterns formed by horizontal and vertical stripe configurations.

## Objectives
- Compare classical, hybrid classical-quantum, and quantum generative models
- Use simulation as a controlled preliminary environment for evaluating the behaviour of models incorporating quantum circuits before deployment on real quantum hardware
- Evaluate generation quality on Bars and Stripes dataset using statistical performance metrics

## Models Implemented
- Classical WGAN-GP
- Hybrid Quantum-Classical WGAN-GP
- Quantum Circuit Born Machine (QCBM)

## Dataset
The project uses the Bars and Stripes (BAS) dataset, consisting of 14 valid binary 3×3 patterns formed by horizontal bars and vertical stripes. The dataset is well suited for quantum simulation due to its low dimensionality while still providing non-trivial generative learning behaviour.

## Evaluation Metrics
- Total Variation Distance
- Support Coverage
- Support Validity
- Empirical Distribution over Valid Patterns

## Repository Structure

## Installation

## Usage

## Results

## Future Work


author: Gabriela Czapska
