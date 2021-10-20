# EnHiC: Learning fine-resolution Hi-C contact maps using a generative adversarial framework


- [EnHiC: Learning fine-resolution Hi-C contact maps using a generative adversarial framework](#enhic-learning-fine-resolution-hi-c-contact-maps-using-a-generative-adversarial-framework)
  - [About](#about)
  - [TODO](#todo)
  - [Setup](#setup)
  - [Data Preparation](#data-preparation)
  - [Traning and Prediction](#traning-and-prediction)
  - [TensorBoard](#tensorboard)
  - [Demo Test](#demo-test)


## About

We develop a new GAN-based model, namely EnHiC, to enhance the resolution of Hi-C contact frequency matrices. Specifically, we propose a novel convolutional layer _Decomposition & Reconstruction Block_ which accounts for the non-negative symmetric property of Hi-C matrices. In our GAN framework, the generator extracts rank-1 features from different scales of low-resolution matrices and predicts the high-resolution matrix via subpixel CNN layers.

> Hu, Yangyang, and Wenxiu Ma. "[EnHiC: learning fine-resolution Hi-C contact maps using a generative adversarial framework](https://academic.oup.com/bioinformatics/article/37/Supplement_1/i272/6319704)." *Bioinformatics* 37.Supplement_1 (2021): i272-i279.

---

## TODO

- [ ] Keep updating the document and cleaning code 
- [x] Fix the format in GitHub markdown
- [ ] Clean and optimize the model
  
---

##  Setup

**Anaconda Pyrhon**

We provide a Conda environment for running EnHiC, use the environment.yml file, which will install all required dependencies:
> conda env create -f environment.yaml

Activate the new environment: 
>conda activate env_EnHiC 

---

##  Data Preparation

As we described in the paper, our model require the input samples are symmeric.
> The EnHiC divides the Hi-C matrix into non-overlapping sub-matrices in size of ![formula](https://render.githubusercontent.com/render/math?math=\frac{n}{2}\times\frac{n}{2}). and then pick out 3 sub-matrices(2 on the diagonal and 1 off diagonal) to combine one sub-matrix in size of ![formula](https://render.githubusercontent.com/render/math?math=n\times{n}). For example, the 2 sub-matrices on the diagonal are ![formula](https://render.githubusercontent.com/render/math?math=(i,i)) and ![formula](https://render.githubusercontent.com/render/math?math=(j,j)) and the off diagonal sub-matrix is ![formula](https://render.githubusercontent.com/render/math?math=(i,j)) so that all ![formula](https://render.githubusercontent.com/render/math?math=n\times{n}) sub-matrices are symmetric. In this section, we select n=400 for EnHiC.

We provide funtions in utils/operations.py， more details please check the demo test.
```
operations.divide_pieces_hic(hic_matrix, block_size=400, max_distance=None, save_file=False, pathfile=None)
operations.merge_hic(hic_lists, index_1D_2D, max_distance=None)
```

---

##  Traning and Prediction
We provide the trained model for data _Rao2014-GM12878-MboI-allreps-filtered.10kb.cool_ at _./pretrained_model/_ with 3 sizes (80, 200, 400)

**Example**
```
from EnHiC import model
gan_model_weights_path = os.path.join('.', 'pretrained_model', 'gen_model_400', 'gen_weights')
Generator = model.make_generator_model(len_high_size=400, scale=4)
Generator.load_weights(gan_model_weights_path)
```

Or you can train your own model.

**Training**
We provide the API function for training data in EnHiC/fit.py
> def train(train_data, valid_data, len_size, scale, EPOCHS, root_path='./', load_model_dir=None, saved_model_dir=None, log_dir=None, summary=False)<br/>
> __train_data__: Tensor in format of tensorflow Dataset (None, len_size, len_size, 1) e.g.(None, 400, 400, 1)<br/>
> __valid_data__: Tensor in format of tensorflow Dataset (None, len_size, len_size, 1) e.g.(None, 400, 400, 1)<br/>
> __len_size__: default: 400. The size of sample must be multiples of 4. e.g. 100, 200, 400.<br/>
> __scale__: the scale of resolution to enhance, e.g. 40kb->10kb: 4, 100kb->10kb: 10<br/>
> __EPOCHS__: number of steps, e.g. 300<br/>
> __root_path__: the directory that to save/load model and log<br/>
> __load_model_dir__: the directory to load exsiting model to continue the training, default is None which means to build a new one <br/>
> __saved_model_dir__: the directory to save model, if it None. The model will be save to root_path/saved_model/gen_model_[len_size]/gen_weights or root_path/saved_model/dis_model_[len_size]/dis_weights <br/>
> __log_dir__: the directory to save model, if it None. The model will be save to root_path/logs <br/>
> __summary__: print the summary of model, default is False<br/>

```
from EnHiC import fit
fit.train(train_data=train_data, valid_data=valid_data, 
          len_size=400, scale=4, EPOCHS=300, 
          root_path='./', load_model_dir=None, saved_model_dir=None, log_dir=None,
          summary=True)
```

**Prediction**
We provide the API function for prediction in EnHiC/fit.py

> def predict(model_path, len_size, scale, ds)
>
>__model_path__: the directory to load exsiting generator model e.g. saved_model/gen_model_[len_size]/gen_weights <br/>
> __len_size__: default: 400. The size of sample must be multiples of 4. e.g. 100, 200, 400. <br/>
> __scale__: the scale of resolution to enhance, e.g. 40kb->10kb: 4, 100kb->10kb: 10 <br/>
> __ds__: Tensor in format of tensorflow Dataset (None, len_size, len_size, 1) e.g.(None, 400, 400, 1) <br/>

```
from EnHiC import fit
mpath = os.path.join(root_dir,'saved_model', 'gen_model_{}'.format(len_size), 'gen_weights')
fit.predict(model_path = mpath, len_size=400, scale=4, ds)
```


## TensorBoard
We log training data and visualize it by [TensorBoard](https://www.tensorflow.org/tensorboard/get_started).
```
tensorboard --logdir=[path-to]/EnHiC/logs/model/
```
or 
```
tensorboard --logdir=[path-to]/EnHiC/logs/model/ --port=${port} --host=${node} --samples_per_plugin images=50
```

---

##  Demo Test

We shows the Demo based on _Rao2014-GM12878-MboI-allreps-filtered.10kb.cool_ (same in our paper, around 1.5Gb)<br/>

**Data preprocessing**
The script _test_preprocessing.py_ prepares the dataset for training. if file doesn't exsit, the script will download it from [MIT Hi-C database](ftp://cooler.csail.mit.edu/coolers/hg19/) to 
_[path-to]/EnHiC/data/raw/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool_

Then the script will call _[path-to]/EnHiC/EnHiC/prepare_data.py_ to divide the Hi-C matrix into samples in the size of ![formula](https://render.githubusercontent.com/render/math?math=(len\_{size}\times{len\_{size}})) within the *genomice_distance*. The samples are saved at 
_[path-to]/EnHiC/data/_ .
__Usage__:
> test_preprocessing.py [chromosome] [len_size] [genomic_distance]
> __chromosome__: the index of chromosome. e.g. 1, 2, 3, ... , 22, X <br/>
> __len_size__: default: 400. The size of sample must be multiples of 4. e.g. 100, 200, 400.<br/>
> __genomic_distance__: default 2000000 (2Mb) <br/>

__Example__:
```
> (env_EnHiC)>> python test_preprocessing.py 1 400 2000000
> (env_EnHiC)>> python test_preprocessing.py 22 400 2000000
```

**Training**
The script _test_train.py_ trains the dataset. The EPOCHS, BATCH_SIZE and chromosome list for training and validation are all configured in the script. It calls _fit.train_ after loading training data.
As a demo, EPOCHS=100, BATCH_SIZE=9, train_chr_list=['22']

__Usage__:
> test_train.py [len_size] [genomic_distance] <br/>
> __len_size__: default: 400. The size of sample must be multiples of 4. e.g. 100, 200, 400. <br/>
> __genomic_distance__: default 2000000 (2Mb) <br/>

__Example__:
```
>> conda activate env_EnHiC
> (env_EnHiC)>> python test_preprocessing.py 22 400 2000000
> (env_EnHiC)>> python test_train.py 400 2000000
```

**Prediction**
The script test_predict.py shows the demo to predict Hi-C low resoltion by EnHiC. 
* Load 10kb Hi-C from cool file
* Downsample 10kb to 40kb
* Divide into samples in the size of $( len\_size \times len\_size)$ within the $genomice\_distance$
* Predict low resolution Hi-C samples
* Combine the samples back into one matrix

  __Usage__:
> test_predict.py [chromosome] [len_size] [genomic_distance] <br/>
> __chromosome__: the index of chromosome. e.g. 1, 2, 3, ... , 22, X<br/>
> __len_size__: default: 400. The size of sample must be multiples of 4. e.g. 100, 200, 400.<br/>
> __genomic_distance__: default 2000000 (2Mb)

__Example__:
```
>> conda activate env_EnHiC
> (env_EnHiC)>> python test_predict.py 22 400 2000000
```

---
