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
> pip list --format=freeze > requirements.txt
>
> pip install -r requirements.txt

## pytorch(GPU) 安装

> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
>
> conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9.2

### For d2l

> conda create --name d2l python=3.8 -y
>
> conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -y
>
> python -c "import torch; print(torch.cuda.is_available())"
>
> curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip

### Jupyter

> conda install -c conda-forge jupyter_contrib_nbextensions
>
> conda install -c conda-forge jupyter_nbextensions_configurator


> jupyter contrib nbextension install --user --skip-running-check
>
> jupyter contrib nbextensions install --user
>
> jupyter nbextensions_configurator enable --user

## PyCharm 远程开发

https://youtrack.jetbrains.com/issue/PY-35978

## 问题

> Q: Error running 'main'
>
> Cannot find remote credentials for target config
>
> A: 重新配置 python 解释器