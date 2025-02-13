
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/zhenyiizhang/DeepRUOT/">
    <img src="figures/logo.svg" alt="Logo" height="150">
  </a>


<h3 align="center">Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport (ICLR'25 oral)</h3>

[Paper Link](https://openreview.net/forum?id=gQlxd3Mtru)

[![Documentation Status](https://readthedocs.org/projects/deepruot/badge/?version=latest)](https://deepruot.readthedocs.io/en/latest/?badge=latest)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-License-green.svg?labelColor=gray)](https://github.com/zhenyiizhang/DeepRUOT/blob/main/LICENSE)
[![commit](https://img.shields.io/github/last-commit/zhenyiizhang/DeepRUOT?color=blue)](https://github.com/zhenyiizhang/DeepRUOT/)

</div>

## Introduction
Reconstructing dynamics using samples from sparsely time-resolved snapshots is an important problem in both natural sciences and machine learning. Here, we introduce a new deep learning approach for solving regularized unbalanced optimal transport (RUOT) and inferring continuous unbalanced stochastic dynamics from observed snapshots. Based on the RUOT form, our method models these dynamics without requiring prior knowledge of growth and death processes or additional information, allowing them to be learnt directly from data. Theoretically, we explore the connections between the RUOT and Schrödinger bridge problem and discuss the key challenges and potential solutions. The effectiveness of our method is demonstrated with a synthetic gene regulatory network, high-dimensional Gaussian Mixture Model, and single-cell RNA-seq data from blood development. Compared with other methods, our approach accurately identifies growth and transition patterns, eliminates false transitions, and constructs the Waddington developmental landscape.

<br />
<div align="left">
  <a href="https://github.com/zhenyiizhang/DeepRUOT/">
    <img src="figures/overview.svg" alt="Logo" height="350">
  </a>

</div>

## Getting Started

1. You can create a new conda environment (DeepRUOT) using

```vim
conda create -n DeepRUOT python=3.10 ipykernel -y
conda activate DeepRUOT
```

2. Install requirements and DeepRUOT
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

The examples can be found in the ```notebook``` directory. Additionally, the model weights required to reproduce the results in the paper can be found in the ```reproduce_model_weights``` directory.

 


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

## Acknowledgments

We thank the following projects for their great work to make our code possible: [MIOFlow](https://github.com/KrishnaswamyLab/MIOFlow/tree/main), [TorchCFM](https://github.com/atong01/conditional-flow-matching).

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