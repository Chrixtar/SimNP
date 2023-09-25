# SimNP: Learning Self-Similarity Priors Between Neural Points

[![arXiv](https://img.shields.io/badge/arXiv-2309.03809-b31b1b.svg)](http://arxiv.org/abs/2309.03809)

Official PyTorch implementation of the paper "SimNP: Learning Self-Similarity Priors Between Neural Points", ICCV 2023.

Christopher Wewer<sup>1</sup>
[Eddy Ilg](https://cvmp.cs.uni-saarland.de/people/#eddy-ilg)<sup>2</sup>
[Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele)<sup>1</sup>
[Jan Eric Lenssen](https://janericlenssen.github.io/)<sup>1</sup>

<sup>1</sup>Max Planck Institute for Informatics, <sup>2</sup>Saarland University


[Project Page](https://geometric-rl.mpi-inf.mpg.de/simnp/) | [Paper](http://arxiv.org/abs/2309.03809) <!-- | [Pretrained models](https://drive.google.com/drive/folders/1OAcwNPxBwaE8aY-0xrHreyP-EWmQYaYJ?usp=sharing) -->

## Requirements

### Install required packages

Make sure you have up-to-date NVIDIA drivers supporting CUDA 11.3

Run

```
conda env create -f environment.yml
conda activate simnp
```

## Usage

1. Clone the repository ```git clone git@github.com:Chrixtar/SimNP.git```.

2. Download the [dataset]<!--(link TBD)-->.

3. Download [pretrained model weights] <!--(link TBD)--> and extract them into the repository.

4. Install requirements ```conda env create -f environment.yml```.

5. (Optional) Setup options in ```options```.

6. (Optional) Run training script with ```python train.py --yaml=[options_path]```.

7. Run inference script with 
```
python predict.py --experiment [path_to_experiment_output_directory] --date [datetime_of_experiment_run]
```, where an experiment is determined by a certain options file and the date is the datetime of the experiment run.

## Citation
```
@inproceedings {wewer2023simnp,
    booktitle = {ICCV},
    title = {SimNP: Learning Self-Similarity Priors Between Neural Points},
    author = {Wewer, Christopher and Ilg, Eddy and Schiele, Bernt and Lenssen, Jan Eric},
    year = {2023},
}
```