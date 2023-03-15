# Pytorch tutorial


## 环境
> python -m venv ./venv

> conda env list
> 
> conda create -n py39 python=3.9
> 
> conda remove -n py36 --all
> 
> conda activate py36
> 
> conda deactivate


> conda env export > environment.yaml
> 
> conda env create -f environment.yaml
> 


## pytorch(GPU) 安装
> conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

## PyCharm 远程开发
https://youtrack.jetbrains.com/issue/PY-35978