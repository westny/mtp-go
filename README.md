# MTP-GO: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural ODEs
> _mtp-go_ is a library containing the implementation for the paper: 
> **MTP-GO: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural ODEs**.
> The paper is currectly available in preprint format on ArXiv and can be accessed [here](https://arxiv.org/abs/2302.00735).
> All code is written using Python 3.10 using a combination of [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).
> Planned contents of this repository will be added incrementally and is underway.


<p align="center">
  <img width="600" src="img/first_page.png">
</p>

##### If you found the content of this repository useful, please consider citing the paper in your work:
```
@article{westny2023graph,
	title="{MTP-GO: G}raph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural {ODE}s",
	author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
	journal={arXiv preprint arXiv:2302.00735},
	year={2023}}
```
***

#### Hardware requirements

The original implementation make use of a considerable amount of data (some gigabytes worth) for training and testing which can be demanding for some setups. For you reference all code has been tried and used on a computer with the following specs:
```
* Processor: Intel® Xeon(R) E-2144G CPU @ 3.60GHz x 8
* Memory: 32 GB
* GPU: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti Rev. A]
```

## Usage

UNDER UPDATE

![](img/model.png)

## Data sets

For model training and evaluation, the [highD](https://www.highd-dataset.com/), [rounD](https://www.round-dataset.com/), and [inD](https://www.ind-dataset.com/) were used.
They are freely available for non-commercial use, but does require that you apply through the links above.

## Preprocessing

Assuming that you have been granted access to any of the above-mentioned data sets, proceed by moving the unzipped content (folder) into a folder named `data_sets` (you have to create this yourself) on the same level as this project. 
The contents may of course be placed in any accessible location of choice but does then require modifications of the preprocessing scripts (see the head of the .py files).

Methods of preprocessing are contained within python scripts. Executing them may be done from a terminal or IDE of choice, for example: 
```bash
python rounD_preprocess.py
```

The output of the preprocessing scripts will be sent to a sub-folder with the name of the data set within the `data` folder. 
Each data sample refers to a traffic sequence and is given a unique index used for easy access. 
A word of **caution**, by this approach, a lot of files are created that could be demanding for some systems.
To make sure unsaved work is not deleted, you will be prompted on your preferred cause of action should the folders already exist: either overwriting existing data or aborting the process.

*Note*: the methods of preprocessing are by no means optimized for speed or computational efficiency.
The process could take several hours depending on the data set and available hardware. 

## License
[Creative Commons](https://creativecommons.org/licenses/by-sa/4.0/)

## Inquiries
> Questions about the paper or the implementations found in this repository should be sent to [_theodor.westny [at] liu.se_](https://liu.se/en/employee/thewe60).
