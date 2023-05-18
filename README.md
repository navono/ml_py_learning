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

> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
> 

### For d2l

> conda create --name d2l python=3.8 -y
> 
> conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
>
> python -c "import torch; print(torch.cuda.is_available())"

## PyCharm 远程开发

https://youtrack.jetbrains.com/issue/PY-35978

## 问题

> Q: Error running 'main'
>
> Cannot find remote credentials for target config
>
> A: 重新配置 python 解释器