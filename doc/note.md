# Planar Reflective Symmetry

##  1. Get Datasets

```
# shapnet
https://www.shapenet.org/download/shapenetcore

# download a synset
http://shapenet.cs.stanford.edu/shapenet/obj-zip/<synsetId>.zip
id = 04225987 # skateboard
```

## 2. Data Processing

- PCL_mesh_sampling

  ![image-20200622011809213](note.assets/image-20200622011809213.png)

  PCL采样不均匀，无法采集完成1000个点，有遗漏

- binvox
  
  将obj转为32\*32\*32的体素，.nrrd格式

- viewvox

![image-20200622012109461](note.assets/image-20200622012109461.png)

![image-20200622154935716](note.assets/image-20200622154935716.png)

![image-20200622154955407](note.assets/image-20200622154955407.png)

## 3. Network Structure

![image-20200623165611424](note.assets/image-20200623165611424.png)

- 5 layers of 3D convolution
- 6 FC sequentials with 3 fully connected layers

### some math
- reflection plane
$$
a_ix+b_iy+c_iz+d_i=0
$$
- quaternion
$$
r_{i0}+r_{i1}\bold{i}+r_{i2}\bold{j}+r_{i3}\bold{k}\\
i^2=j^2=k^2=-1\\
ij=k、ji=-k、jk=i、kj=-i、ki=j、ik=-j
$$

### Loss

- Symmetry Distance Loss
  
  $$
  L_{sd} = \displaystyle\sum_{i=k}^ND_k
  $$
- Regularization Loss
  $$
  L_r = \displaystyle\sum_{i=1}^3\sum_{j=1}^3
  (A_{ij}^2 + B_{ij}^2)
  $$
- Overall Loss Funtion
  $$
  L = L_{sd}+\omega_rL_r
  $$

Reference：

1. [pytorch入门](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)
2. [binvox使用](https://web.archive.org/web/20131213132022/https://minecraft.gamepedia.com/Binvox)
3. [Another implementation of PRS_Net](https://github.com/Shanmwy/PRS-Net)

