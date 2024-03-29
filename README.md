# NNSplitter 

Welcome to the NNSplitter project! This repository contains the core code for NNSplitter, a tool that actively protects the DNN model by splitting it into two parts: the obfuscated model that performs poorly due to weight obfuscation, and the model secrets consisting of the indexes and original values of the obfuscated weights, which can only be accessed by authorized users with the support of the trusted execution environment (TEE）. Please note that our method can be applied to any pre-trained models and this repository does not include any specific one. However, you can find the model with pre-trained weights on open-sourced GitHub projects (for example, https://github.com/huyvnphan/PyTorch_CIFAR10). For details of this work, please refer to the paper available at https://proceedings.mlr.press/v202/zhou23h.html.

## Description of TEE interaction

In the original paper, we store the index and original value of the obfuscated weights as the model secrets. However, to simplify the computation process for normal users from the implementation perspective,  **it's more efficient to store the weight changes (∆W') and their filter index** instead. In this case, for a specific layer, we compute the convolution of the obfuscated weights (W+∆W') and input features ($X_i$) in the normal worldThis yields an output feature map ($O_n$) containing errors in certain output channels. Simultaneously, within a secure computational domain, we compute the convolution of input features with ∆W' (denoted as $O_s$). These secure computations are subtracted from $O_n$ to correct errors in specific output channels, thereby obtaining benign features for regular users (i.e., $O_s - O_w = W * X_i := X_{i+1}$, by virtue of the distributive nature of the convolution operation).  Now the incorrect output feature map is corrected in the secure world. For the next layer, the normal world program directly calls $X_{i+1}$ from the secure world and fetches the obfuscated model for computation, then the output feature maps with errors are transmitted to secure memory for correction as in the previous layer. 

Moreover, if attackers are malicious users capable of observing the feature maps for authorized inference, feature encryption can be applied to safeguard $X_{i+1}$ and further bolster security, thereby preventing leakage of benign feature maps (as outlined in [1]). It's important to note that this scenario differs from our established threat model, where we do not assume malicious users as attackers. Hence, attackers cannot observe the process of authorized inference in our work. However, we mention this scenario here for the sake of discussing potential attackers with enhanced capabilities and exploring possible countermeasures (there could be more efficient solutions to defend against such stronger attackers, e.g., reducing latency overhead due to feature encryption for normal users, and we leave it for future studies).

[1] No Privacy Left Outside: On the (In-)Security of TEE-Shielded DNN Partition for On-Device ML. S&P, 2024.

## Requirements

Before getting started, make sure you have the following dependencies installed:

- Python 3.6
- torch 1.10

## Code Structure

The core code of NNSplitter is organized into the following files:

- `controller_rnn.py`: This file contains the implementation of a recurrent neural network (RNN) controller that utilizes the REINFORCE algorithm to generate desired parameters.

- `train.py`: This script is used to train the victim model with the goal of reducing its accuracy.

- `main_cifar.py`: In this file, you can optimize the victim model using the parameters generated by the controller.

- `utilis.py`: Here, you will find various helper functions that assist in the processing of text splitting.

Feel free to explore the code and modify it according to your needs.

## Getting Started

To start using NNSplitter, follow these steps:

1. Ensure that you have met the requirements mentioned above.

2. Obtain the pre-trained model with weights and change the model import path.

3. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Tongzhou0101/NNSplitter

## Cite

If you find this work or code helpful, please cite us:

```bibtex
@InProceedings{pmlr-v202-zhou23h,
  title = 	 {{NNS}plitter: An Active Defense Solution for {DNN} Model via Automated Weight Obfuscation},
  author =       {Zhou, Tong and Luo, Yukui and Ren, Shaolei and Xu, Xiaolin},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {42614--42624},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR}
}
```
