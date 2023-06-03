# Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Differentially Constrained Motion Models

> ### Updates
>  *April 2023* :date:
> - Update repository to include functionality to reproduce paper 2.
> - Migrate code to torch==2.0.0. Update requirements.

> ### Description
> _mtp-go_ is a library containing the implementation for the papers: 
> 1. *MTP-GO: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural ODEs* ([ArXiv:page_with_curl:](https://arxiv.org/abs/2302.00735)), accepted for IEEE Transactions on Intelligent Vehicles ([TIV](https://ieeexplore.ieee.org/document/10143287)).
> 2. *Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction* ([ArXiv:page_with_curl:](https://arxiv.org/abs/2304.05116)), accepted for IEEE 2023 Intelligent Vehicles Symposium ([IV2023](https://2023.ieee-iv.org/)).
> 
> Both papers are available in preprint format on ArXiv by the links above.
> All code is written using Python 3.11 using a combination of [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

<p align="center">
  <img width="600" src="img/first_page.png">
</p>

##### If you found the content of this repository useful, please consider citing the papers in your work:
```
@article{westny2023mtpgo,
  title="{MTP-GO}: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural {ODEs}",
  author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2023},
  volume={},
  number={},
  pages={},
  doi={10.1109/TIV.2023.3282308}}
} 
```

```
@inproceedings{westny2023eval,
  title={Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction},
  author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
  booktitle={IEEE Intelligent Vehicles Symposium (IV)},
  pages={},
  year={2023},
}
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

Most of the necessary building blocks to implement MTP-GO is contained within the `models` folder. 
The main files of interest are:
- [gru_gnn.py](models/gru_gnn.py)
- [motion_models.py](models/motion_models.py)
- [base_mdn.py](base_mdn.py)

In `gru_gnn.py` the complete encoder-decoder model implementation is contained.
This includes a custom GRU cell implementation that make use of layers based on Graph Neural Networks.

In `motion_models.py` the implementations of the various motion models are contained, including the neural ODEs, used to learn road-user differential constraints. 
This is also where you will find functions used to perform the Jacobian calculations of the model.

In this work, [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/) was used to implement the training and testing behavior.
Since most of the functionality is still implemented using pytorch, you are not restricted to using lightning, but it is recommended given the additional functionality.
In `base_mdn.py` the lightning-based abstraction of MTP-GO is contained.
This module is used to implement batch-wise forward and backward passes as well as to specify training and testing behavior.

Assuming data is available, training a model based on MTP-GO *is as easy* as running `train.py` in an environment with the necessary libraries installed, e.g.:
```bash
python train.py --dataset rounD --motion-model neuralode --n-workers 8 --hidden-size 128
```
To learn more about the objective-scheduling algorithm described in the paper as well as the loss functions used, see [losses.py](losses.py).

![Schematics of MTP-GO](img/model.png)

## Data sets

For model training and evaluation, the [highD](https://www.highd-dataset.com/), [rounD](https://www.round-dataset.com/), and [inD](https://www.ind-dataset.com/) were used. The data sets contain recorded trajectories from different locations in Germany, including various highways, roundabouts, and intersections. The data includes several hours of naturalistic driving data recorded at 25 Hz of considerable quality.
They are freely available for non-commercial use, but does require applying for usage through the links above.

![round2](https://user-images.githubusercontent.com/60364134/220960422-4e7d7e13-c9b3-42af-99d3-a61eb6406e1e.gif)


## Preprocessing

Assuming that you have been granted access to any of the above-mentioned data sets, proceed by moving the unzipped content (folder) into a folder named `data_sets` (you have to create this yourself) on the same level as this project. 
The contents may of course be placed in any accessible location of choice but does then require modifications of the preprocessing scripts (see the head of the .py files).

Methods of preprocessing are contained within python scripts. Executing them may be done from a terminal or IDE of choice **(from within this project folder)**, for example: 
```bash
python rounD_preprocess.py
```

The output of the preprocessing scripts will be sent to a sub-folder with the name of the data set within the `data` folder in this project. 
Each data sample refers to a traffic sequence and is given a unique index used for easy access. 

:exclamation: A word of **caution**, by this approach, a lot of files are created that could be demanding for some systems.

To make sure unsaved work is not deleted, you will be prompted on your preferred cause of action should the folders already exist: either overwriting existing data or aborting the process.

:triangular_flag_on_post: The methods of preprocessing are by no means optimized for speed or computational efficiency.
The process could take several hours depending on the data set and available hardware. 

## License
[Creative Commons](https://creativecommons.org/licenses/by-sa/4.0/)

## Inquiries
> Questions about the paper or the implementations found in this repository should be sent to [_theodor.westny [at] liu.se_](https://liu.se/en/employee/thewe60).
