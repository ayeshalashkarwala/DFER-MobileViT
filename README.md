# [**Driver Facial Expression Recognition using Global Cotext Vision Transformers**](https://ieeexplore.ieee.org/document/10464794) : MobileViT Extension

This repository contains the research project conducted at **Brandenburg University of Technology (BTU) Cottbus-Senftenberg**. The study evaluates the performance and efficiency of lightweight Vision Transformers for real-world driver monitoring systems.

## Acknowledgments
This project builds upon the foundational work and repository provided by **Ibtissam Saadi** for the paper: 
*"Shuffle Vision Transformer: Lightweight, Fast and Efficient Recognition of Driver’s Facial Expression"*.

## Project Overview
The objective of this extension was to analyze the trade-off between model complexity and performance by replacing the Global Context Vision Transformer (GC-ViT) with **MobileViT**. The goal was to determine if lightweight architectures can achieve sufficient accuracy for driver safety applications while significantly reducing computational overhead.

### Experimental Results
The following metrics were recorded using the KMU-FED dataset:

| Metric | Global ViT (gc-vit-t) | Mobile ViT (mobilevit_s) |
| :--- | :--- | :--- |
| **Parameters** | 28 Million | 5.6 Million |
| **Inference Time** | 11h 56m 49s | 9h 7m 5s |
| **Accuracy** | 1.0000 | 0.9090 |
| **F1-Score** | 0.9812 | 0.9042 |

### Key Findings
* The implementation of MobileViT achieved an 80% reduction in total parameters.
* Total inference time was reduced by nearly 3 hours across the dataset.
* While a slight decrease in accuracy was observed compared to GC-ViT, the MobileViT model maintains a high F1-score (0.9042), demonstrating its suitability for edge deployment.

---

## Datasets
* **KMU-FED**: [https://cvpr.kmu.ac.kr/KMU-FED.html](https://cvpr.kmu.ac.kr/KMU-FED.html)

## Implementation Steps

### Preprocessing
* **For KMU-FED**: Run `python preprocess_kmu.py` to save data in .h5 format, then use `KMU.py` to split data into 10 folds.

### Training and Testing
* **KMU-FED (10-fold Cross-Validation)**: `python 10fold.py`

---

## Contact
**Author:** Ayesha Fazal Lashkarwala 

**Institution:** Brandenburg University of Technology (BTU) Cottbus-Senftenberg  

**Supervisor:** Ibtissam Saadi, Douglas W. Cunningham
