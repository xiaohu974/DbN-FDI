# DbN-FDI
- 《Integrating Spatial and Frequency Domain Information via a Dual-Branch Network for Enhanced Low-Light Endoscopic Imaging》.
# Training
- Download the [Endo4IE](https://data.mendeley.com/datasets/3j3tmghw33/1)
- Use the below command for training:
```
python train.py
```
# Testing
- Use the below command for testing:
```
python evaluation.py
```
# data_loaders 
- To mitigate Fourier transform’s sensitivity to rotated images, we apply random horizontal/vertical flipping for data augmentation.
# Project Environment
- Use the below command for install project enviroment:
```
pip install requirements.txt
```
# Requirement
```
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
attrs==21.4.0
backcall==0.2.0
beautifulsoup4==4.11.1
bleach==5.0.1
catboost==1.0.6
certifi @ file:///opt/conda/conda-bld/certifi_1655968806487/work/certifi
cffi==1.15.1
charset-normalizer==2.1.0
cloudpickle==2.1.0
complexPyTorch==0.4
cycler==0.11.0
debugpy==1.6.2
decorator==5.1.1
defusedxml==0.7.1
einops==0.6.1
entrypoints==0.4
fastjsonschema==2.15.3
filelock==3.12.0
fonttools==4.34.4
graphviz==0.20
grpcio==1.47.0
h5py==3.7.0
huggingface-hub==0.13.4
idna==3.3
imageio==2.31.1
ipykernel==6.15.1
ipython==7.34.0
ipython-genutils==0.2.0
ipywidgets==7.7.1
IQA-pytorch==0.1
jedi==0.18.1
Jinja2==3.1.2
joblib==1.1.0
jsonschema==4.7.2
jupyter-client==7.3.4
jupyter-core==4.11.1
jupyterlab-pygments==0.2.2
jupyterlab-widgets==1.1.1
kiwisolver==1.4.3
lazy_loader==0.2
lightgbm==3.3.2
MarkupSafe==2.1.1
matplotlib==3.5.2
matplotlib-inline==0.1.3
mistune==0.8.4
natsort==8.3.1
nbclient==0.6.6
nbconvert==6.5.0
nbformat==5.4.0
nest-asyncio==1.5.5
networkx==3.1
notebook==6.4.12
numpy==1.23.1
opencv-python==4.6.0.66
packaging==21.3
pandas==1.4.3
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==9.2.0
plotly==5.9.0
prometheus-client==0.14.1
prompt-toolkit==3.0.30
psutil==5.9.1
ptyprocess==0.7.0
pycparser==2.21
Pygments==2.12.0
pyparsing==3.0.9
pyrsistent==0.18.1
python-dateutil==2.8.2
pytz==2022.1
PyWavelets==1.4.1
PyYAML==6.0
pyzmq==23.2.0
requests==2.28.1
scikit-image==0.21.0
scikit-learn==1.1.1
scipy==1.8.1
seaborn==0.11.2
Send2Trash==1.8.0
six==1.16.0
sklearn==0.0
soupsieve==2.3.2.post1
spyder-kernels==2.3.2
tenacity==8.0.1
terminado==0.15.0
threadpoolctl==3.1.0
tifffile==2023.4.12
timm==0.6.13
tinycss2==1.1.1
torch==1.12.0+cu113
torchaudio==0.12.0+cu113
torchvision==0.13.0+cu113
tornado==6.2
tqdm==4.64.0
traitlets==5.3.0
typing_extensions==4.3.0
urllib3==1.26.10
wcwidth==0.2.5
webencodings==0.5.1
widgetsnbextension==3.6.1
wurlitzer==3.0.2
xgboost==1.6.1
```
