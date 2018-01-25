## Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction
Chen-Hsuan Lin, Chen Kong, and Simon Lucey  
AAAI Conference on Artificial Intelligence (AAAI), 2018  

Paper: https://www.andrew.cmu.edu/user/chenhsul/paper/AAAI2018.pdf  
arXiv preprint: https://arxiv.org/abs/1706.07036

We provide TensorFlow code for the single-category experiment.

--------------------------------------

### Prerequisites  
This code is developed with Python3 (`python3`). TensorFlow r1.0+ is required. The dependencies can install by running
```
pip3 install --upgrade numpy scipy termcolor tensorflow-gpu
```
If you don't have sudo access, add the `--user` flag.  

### Dataset  
The dataset can be downloaded [here](https://cmu.box.com/s/luwzlzhrn8o5ps6oiwawn5mj44sqp2fl ) (7.4GB). This file includes:
- Ground-truth point cloud of ShapeNet chair CAD models (densified to 100K points)
- Pre-rendered depth images for training
- Input RGB images (from [Perspective Transformer Nets](https://github.com/xcyan/nips16_PTN))

Please also cite the relevant papers if you plan to use this dataset package.

### Running the code  
(to be updated soon)

--------------------------------------

If you find our code useful for your research, please cite
```
@inproceedings{lin2018learning,
  title={Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction},
  author={Lin, Chen-Hsuan and Kong, Chen and Lucey, Simon},
  booktitle={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2018}
}
```

Please contact me (chlin@cmu.edu) if you have any questions!


