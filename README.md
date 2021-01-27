# EnHiC: Learning fine-resolution Hi-C contact maps using a generative adversarial framework

---

- [EnHiC: Learning fine-resolution Hi-C contact maps using a generative adversarial framework](#enhic-learning-fine-resolution-hi-c-contact-maps-using-a-generative-adversarial-framework)
  - [About](#about)
  - [Setup](#setup)
  - [Data Preparation](#data-preparation)
  - [Traning and Prediction](#traning-and-prediction)
  - [Demo Test](#demo-test)

---

## About

---

##  Setup

**Anaconda Pyrhon**

We provide a Conda environment for running EnHiC, use the environment.yaml file, which will install all required dependencies:
> conda env create -f environment.yaml

Activate the new environment: 
>conda activate env_EnHiC 

---

##  Data Preparation

---

##  Traning and Prediction

---

##  Demo Test
**Data preprocessing**
Example:
> (env_EnHiC)>> python test_preprocessing.py 1 200 2000000
> (env_EnHiC)>> python test_preprocessing.py 22 200 2000000

**Training**
Example:
> (env_EnHiC)>> python test_train.py 200 2000000

**Prediction**

Example:
> (env_EnHiC)>> python test_predict.py 22 200 2000000

---