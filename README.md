# Evolving GraRep: Unsupervised Learning Adaptation and Memory Efficient Design with UDR-GraRep and Tiny-GraRep

## Overview
This project implements the UDR-GraRep and Tiny-GraRep algorithms for graph representation learning. The codebase includes scripts to generate representations and evaluate their performance.

## Getting Started

### Prerequisites
- python3

### Installation
- pip install -r requirements.txt

## Usage

To use this project, follow these steps:

1. **Generate Representations:**
   - Run the `generate_reps.py` script to generate the graph representations. 
     ```
     python3 generate_reps.py
     ```

2. **Run Experiments:**
   - After generating the representations, execute `Experiment.py` to evaluate the performance of UDR-GraRep and Tiny-GraRep.
     ```
     python3 Experiment.py
     ```
   - For the TinyGraRep speed and memory size study, execute `tiny_vs_legacy.py`. Make sure the embeddings and tmp folders are empty or deleted for all datasets.
     ```
     python3 tiny_vs_legacy.py
     ```


## Contact
- Yuhang Wang:  wyh5699@gmail.com
- Ruben Ahrens: s3677532@vuw.leidenuniv.nl



