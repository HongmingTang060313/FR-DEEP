# FR DEEP

** this is still under construction, please be aware of this **

FR DEEP is a dataset of labeled radio galaxies used for deep learning training. It's a subset of the training data of 'Transfer Learning for Radio Galaxy Classification', where data in the work experienced data augmentation based on FR DEEP. These labeled samples came from FRICAT and CoNFIG catalogs. Each considered sample has been classified as either FR I or FR II radio source with certainty. If you use the dataset please cite arXiv:1903.11921 [astro-ph.IM].

** Transfer learning for radio galaxy classification **
Tang H., Scaife A.~M.~M., Leahy J.~P., 2019, arXiv e-prints, arXiv:1903.11921 (https://arxiv.org/abs/1903.11921)  

# The FR DEEP Batched Dataset

The FR DEEP batched dataset comprises two separate sub-datasets: NVSS set and FIRST set. The two sub sets adopted same objects from the two aforementioned catalogs. Each sub set contains 600 150 x 150 images in two class: FR I and FR II(264 FR I and 336 FR II objects). Sample images are downloaded via Skyview Virtual Observatory, and experienced:

(1) sigma-clipping 

(2) image centred crop {300 x 300 pixels to 150 x 150 pixels}

(3) give FR I object label {0}, and FR II source label {1}


There are 550 training images, and 50 test images. The FR DEEP dataset is inspired by [CIFAR-10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html).

The dataset is divided by 11 training batches and 1 testing batch. Each batch has 50 images. In every batch, 22 FR I and 34 FR II objects images are organized in random order.

This is an *imbalance dataset*

NVSS set images are looks like:

FR I: ![a](/media/nvss/FR1/1_CoNFIG_FR1_.png) ![b](/media/nvss/FR1/1_FRICAT_FR1_.png) ![c](/media/nvss/FR1/2_FRICAT_FR1_.png) ![d](/media/nvss/FR1/2_CoNFIG_FR1_.png)

FR II: ![a](/media/nvss/FR2/53_CoNFIG_FR2.png) ![b](/media/nvss/FR2/54_CoNFIG_FR2.png) ![c](/media/nvss/FR2/55_CoNFIG_FR2.png) ![d](/media/nvss/FR2/56_CoNFIG_FR2.png)

FIRST set images, on the other hand, are like:

FR I: ![a](/media/first/FR1/2_CoNFIG_FR1.png) ![b](/media/first/FR1/2_FRICAT_FR1.png) ![c](/media/first/FR1/3_FRICAT_FR1.png) ![d](/media/first/FR1/3_CoNFIG_FR1.png)

FR II: ![a](/media/first/FR2/11_CoNFIG_FR2.png) ![b](/media/first/FR2/12_CoNFIG_FR2.png) ![c](/media/first/FR2/13_CoNFIG_FR2.png) ![d](/media/first/FR2/14_CoNFIG_FR2.png)

## Using the Dataset in PyTorch

The [htru2.py] and [htru3.py] file contains an instance of the [torchvision Dataset()](https://pytorch.org/docs/stable/torchvision/datasets.html) for the FR DEEP Batched Dataset. 

To use it with PyTorch in Python, first import the torchvision datasets and transforms libraries:

```python
from torchvision import datasets
import torchvision.transforms as transforms
```

Then import the HTRU1 class:

```python
from htru1 import HTRU1
```

Define the transform:

```python
# convert data to a normalized torch.FloatTensor
transform = transforms.Compose(
    [transforms.Lambda(lambda x: select_channel(x,0,'grey')), # 'RGB' in the context of htru1
     transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])])
 ```

Read the HTRU1 dataset from '/NVSS_data':

```python
# choose the training and test datasets
train_data = HTRU1('/NVSS_data', train=True, download=False, transform=transform)
test_data = HTRU1('/NVSS_data', train=False, download=False, transform=transform)
```

### Using Individual Channels in PyTorch

If you want to use only one of the "channels" in the HTRU1 Batched Dataset, you can extract it using the torchvision generic transform [transforms.Lambda](https://pytorch.org/docs/stable/torchvision/transforms.html#generic-transforms). 

This function extracts a specific channel ("c") and writes the image of that channel out as a greyscale PIL Image:

```python
def select_channel(x,c,color=None):
    
    from PIL import Image
    
    np_img = np.array(x, dtype=np.uint8)
    if color=='RGB':
        ch_img = np_img[:,:,c]
    elif color=='grey':
        ch_img = np_img
    img = Image.fromarray(ch_img,'L')
    return img
 ```
 
### Jupyter Notebooks

An example of classification using the HTRU1 class in PyTorch is provided as a Jupyter notebook [extracting an individual channel as a greyscale image](https://github.com/as595/HTRU1/blob/master/htru1_tutorial_channel.ipynb).




