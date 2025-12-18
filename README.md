# Slicing Adversarial Networks (SAN) – Deep Learning Project

This repository contains our implementation and experimental study of **Slicing Adversarial Networks (SAN)**, a GAN variant designed to improve training stability and enforce metrizability by explicitly optimizing the projection direction in the discriminator.

This project is **based on the paper**:

Takida, Yuhta et al., *SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer*,  
Proceedings of the Twelfth International Conference on Learning Representations (ICLR), 2024.  
https://openreview.net/forum?id=eiF7TU1E8E

The project was developed as part of a deep learning course and was ranked  **first amongst the entire class (1/30)** in termes of precision and recall. 

---

## Project Overview

Generative Adversarial Networks (GANs) are powerful generative models but often suffer from instability and lack a well-defined distance between the generated and real data distributions.

**SAN (Slicing Adversarial Network)** addresses this issue by:
- decomposing the discriminator into a feature extractor and a direction on the hypersphere,
- explicitly optimizing this direction to satisfy direction optimality,
- improving precision, recall, and overall sample quality.

---

## Results

We were ranked **1/30** in termes of precision and recall. 
Where our best model achieved the following performance:

- **FID**: 31.69  
- **Precision**: 0.81  
- **Recall**: 0.57  

These correspond to the **best precision and recall** obtained amongst **the entire class**.

*These scores are not optimal because they correspond to a simple type of GAN architecture that we used only to show the effectiness of the methdo. We could further improve the performances by using convolutional layers rather than linear.* 

---

## Best Model Configuration

The most optimal model we retained is a **SAN** trained with:

- Optimizer: Adam  
- Batch size: 128  
- Learning rate: 0.0001  
- Epochs: 300  

---

## Repository Structure

- `train.py` – Training script  
- `model.py` – Generator and discriminator architectures (SAN implementation)  
- `utils.py` – Utility functions including the training functions and metrics 
- `GAN/` and `SAN/` – Model weights 
- `checkpoints/` – Best model's weights
- `visualize.py` – Function to plot the CDF


Additional material:
- `slides.pdf` – Visual results, figures, and training dynamics  
- `report.pdf` – Full report with theoretical background, methodology, and experiments  

---

## generate.py

Use the file `generate.py` to generate **10,000 MNIST samples** in the `samples/` folder.

Example:
  > python3 generate.py --bacth_size 64

## References

```bibtex
@inproceedings{takida2024san,
  title={{SAN}: Inducing Metrizability of {GAN} with Discriminative Normalized Linear Layer},
  author={Takida, Yuhta and Imaizumi, Masaaki and Shibuya, Takashi and Lai, Chieh-Hsin and Uesaka, Toshimitsu and Murata, Naoki and Mitsufuji, Yuki},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=eiF7TU1E8E}
}




## generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. 
Example:
  > python3 generate.py --bacth_size 64

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt


## Checkpoints
Push the minimal amount of models in the folder *GAN* or *SAN*.
