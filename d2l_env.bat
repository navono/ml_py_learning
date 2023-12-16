@ECHO OFF

set BASEDIR=%~dp0
PUSHD %BASEDIR%
echo %BASEDIR%

set envName=d2l

call conda env list | findstr /C:%envName% > nul
if %errorlevel% equ 0 (
  echo %envName% environment exists. Activating it...
  call conda activate %envName%
) else (
  echo %envName% environment does not exist. Creating it...
  call conda create --name %envName% python=3.9 -y
  call conda activate %envName%
)

echo Installing dependencies...
pip install torch==1.12.0
pip install torchvision==0.13.0
pip install d2l==0.17.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -y

POPD
