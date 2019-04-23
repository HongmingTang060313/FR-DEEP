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

