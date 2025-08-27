
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/zhenyiizhang/DeepRUOT/">
    <img src="figures/logo.png" alt="Logo" height="150">
  </a>


<h3 align="center">Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport (ICLR'25 oral)</h3>

[Paper Link](https://openreview.net/forum?id=gQlxd3Mtru)

[![Documentation Status](https://readthedocs.org/projects/deepruot/badge/?version=latest)](https://deepruot.readthedocs.io/en/latest/?badge=latest)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-License-green.svg?labelColor=gray)](https://github.com/zhenyiizhang/DeepRUOT/blob/main/LICENSE)
[![commit](https://img.shields.io/github/last-commit/zhenyiizhang/DeepRUOT?color=blue)](https://github.com/zhenyiizhang/DeepRUOT/)

</div>

## Updated
**(2025/08/26)** In [DeepRUOTv2](https://github.com/zhenyiizhang/DeepRUOTv2) version (https://github.com/zhenyiizhang/DeepRUOTv2), we conduct the following updates:
* Added a new `use_mass` option to the configuration file. Setting this to `False` disables the growth term.
* The `sigma` parameter can now be set to `0.0` to disable stochastic effects.
* The calculated results can now be automatically evaluated.
* Added a new Notebook for downstream analysis at `evaluation/analysis.ipynb`. This script enables more advanced analyses, including data interpolation, inferring fate probabilities, and further gene-level studies ([view tutorial](https://deepruot.readthedocs.io/en/latest/notebook/analysis.html)).

Based on the [DeepRUOTv2](https://github.com/zhenyiizhang/DeepRUOTv2) (https://github.com/zhenyiizhang/DeepRUOTv2) framework, we implement a suite of state-of-the-art dynamical optimal transport methods:

- **Dynamical Optimal Transport (OT)**: classical formulation without growth/death processes or stochastic effects.
- **Unbalanced Dynamical OT**: extension of dynamical OT that accounts for growth and death processes, but without stochastic effects.
- **Dynamical Schrödinger Bridge (SB)**: stochastic extension of dynamical OT without growth/death.
- **Regularized Unbalanced Optimal Transport (Unbalanced Schrödinger Bridge)**: general formulation that incorporates both growth/death processes and stochasticity.

Users can flexibly specify the desired model through configuration files to select the appropriate solver for their application. Furthermore, some downstream analysis can be conducted ([view tutorial](https://deepruot.readthedocs.io/en/latest/notebook/analysis.html)).

**(2025/06/01)** We now release an updated version (v2) of DeepRUOT, which includes computations on more scRNA datasets presented in our latest work (https://arxiv.org/abs/2505.11197): Mouse Blood Hematopoiesis (50D), Embryoid Body (50D), Pancreatic $\beta$ -cell differentiation (30D) and  A549 EMT (10D). We warmly welcome everyone to use the [DeepRUOTv2](https://github.com/zhenyiizhang/DeepRUOTv2) version (https://github.com/zhenyiizhang/DeepRUOTv2).

## Introduction
Reconstructing dynamics using samples from sparsely time-resolved snapshots is an important problem in both natural sciences and machine learning. Here, we introduce a new deep learning approach for solving regularized unbalanced optimal transport (RUOT) and inferring continuous unbalanced stochastic dynamics from observed snapshots. Based on the RUOT form, our method models these dynamics without requiring prior knowledge of growth and death processes or additional information, allowing them to be learnt directly from data. Theoretically, we explore the connections between the RUOT and Schrödinger bridge problem and discuss the key challenges and potential solutions. The effectiveness of our method is demonstrated with a synthetic gene regulatory network, high-dimensional Gaussian Mixture Model, and single-cell RNA-seq data from blood development. Compared with other methods, our approach accurately identifies growth and transition patterns, eliminates false transitions, and constructs the Waddington developmental landscape.

<br />
<div align="left">
  <a href="https://github.com/zhenyiizhang/DeepRUOT/">
    <img src="figures/overview.png" alt="Logo" height="350">
  </a>

</div>

## Getting Started

1. Clone this repository:

```vim
git clone https://github.com/zhenyiizhang/DeepRUOT
```

2. You can create a new conda environment (DeepRUOT) using

```vim
conda create -n DeepRUOT python=3.10 ipykernel -y
conda activate DeepRUOT
```

3. Install requirements and DeepRUOT
```vim
cd path_to_DeepRUOT
pip install -r requirements.txt
pip install -e .
```

## How to use

Please check the [Tutorials](https://deepruot.readthedocs.io/en/latest/index.html), where we provide four examples:
- Gene Regulatory Network Simulation
- Mouse Hematopoiesis scRNA Data
- Epithelial-to-Mesenchymal Transition (EMT) Data
- Gaussian Mixture Data (20D)

The examples can be found in the ```notebook``` directory. Additionally, the model weights required to reproduce the results in the paper can be found in the ```reproduce_model_weights``` directory. Both CUDA-enabled GPU and Mac MPS are supported for computation.


## Contact information

- Zhenyi Zhang (SMS, PKU)-[zhenyizhang@stu.pku.edu.cn](mailto:zhenyizhang@stu.pku.edu.cn)
- Peijie Zhou (CMLR, PKU) (Corresponding author)-[pjzhou@pku.edu.cn](mailto:pjzhou@pku.edu.cn)
- Tiejun Li (SMS & CMLR, PKU) (Corresponding author)-[tieli@pku.edu.cn](mailto:tieli@pku.edu.cn)

## How to cite

If you find DeepRUOT useful in your research, please consider citing our work.

<details>
<summary>
Zhang, Z., Li, T., & Zhou, P. (2025). Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport. The Thirteenth International Conference on Learning Representations.
</summary>

```bibtex
@inproceedings{
zhang2025learning,
title={Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport},
author={Zhenyi Zhang and Tiejun Li and Peijie Zhou},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=gQlxd3Mtru}
}
```
</details>

### Want to know more?

Check out our latest review on modeling cellular dynamics! We explore how scRNA-seq, spatial transcriptomics, and advanced computational tools like Markov chains & generative models reveal spatiotemporal cell trajectories and fate decisions!

[https://doi.org/10.3390/e27050453](https://doi.org/10.3390/e27050453)


<br />
<div align="center">
  <a href="https://github.com/zhenyiizhang/DeepRUOT/">
    <img src="figures/Review_dyn.png" alt="Logo" height="300">
  </a>

</div>

## Acknowledgments

We thank the following projects for their great work to make our code possible: [MIOFlow](https://github.com/KrishnaswamyLab/MIOFlow/tree/main), [TorchCFM](https://github.com/atong01/conditional-flow-matching). We are also grateful for the exciting work in trajectory inference, which has greatly inspired and influenced our work.

## License
DeepRUOT is licensed under the MIT License, and the code from MIOflow used in this project is subject to the Yale Non-Commercial License.

```
License

Copyright (c) 2025 Zhenyi Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The code from MIOflow used in this project is subject to the Yale Non-Commercial License.

```