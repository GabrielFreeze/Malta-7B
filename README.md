# Malta-7B
Malta-7B is a news article search and comprehension tool for Maltese News Articles.

![screenshot](https://github.com/GabrielFreeze/Malta-7B/assets/81355262/99c7d480-75ba-49fd-bade-ff2eebb28689)


## Install instructions

### Step 1 - Create Virtual Environment
It is recommended that you create a `conda` environment to avoid dependency issues
```bash
conda create -n malta-7B python=3.8 -y
conda activate malta-7B
conda install pip
```

### Step 2 - Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 - Install CUDA
<ul>
  <li><a href="https://pytorch.org/get-started/locally/" target="_blank">Install PyTorch w/ CUDA support</a></li>
  <li><a href="https://developer.nvidia.com/cuda-downloads" target="_blank">Install appropriate CUDA toolkit</a></li>
  <li><a href="https://developer.nvidia.com/cuda-downloads](https://medium.com/@harunijaz/a-step-by-step-guide-to-installing-cuda-with-pytorch-in-conda-on-windows-verifying-via-console-9ba4cd5ccbef" target="_blank">CUDA Installation Tutorial</a></li>
</ul>

```bash
#Check Installation
python -c "import torch; print(torch.cuda.is_available())" #True
```

## Usage

### Step 1 - Data Directory
Place your news articles under ```data/data``` in the format ```data_{year}.csv```. Refer to [test.csv](https://github.com/GabrielFreeze/Malta-7B/blob/main/data/data/test.csv) for column format.


### Step 2 - Change Directory
```bash
cd malta-7b/src
```

### Step 3 - Create Vector Databases
Supply appropriate date range according to provided ```.csv``` files.
```bash
python vectorize_docs.py 2018 2024
```


### Step 4 - Launch System
Access the grad.io link printed to standard output.
```bash
python main.py
```
