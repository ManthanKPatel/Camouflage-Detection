## Implement deep learning approach to find and segment camouflaged objects in a scene.
### Applications:
1. Nature expedition
2. Military Surveillance
3. Polyp Detection

### Datasets Used:

#### 1. Training Dataset - COD10K 

<img src="https://user-images.githubusercontent.com/90741568/234659401-ed68f03c-c435-4c2d-925a-b6f7ca651d79.png" width="300" height="300">

#### 2. Test Dataset


##### i. MoCA video Dataset
<img src="https://user-images.githubusercontent.com/90741568/234661986-c3a7c764-f9ab-464c-8170-55981be442a4.png" width="300" height="200"> 

##### ii. Military Personnel Dataset
<img src="https://user-images.githubusercontent.com/90741568/234663513-e90d9702-646c-4a46-887e-058a371108cb.png" width="300" height="200">

### Overview
<img width="843" alt="SINet" src="https://user-images.githubusercontent.com/90741568/234704629-efdfe39c-89e5-4371-992b-82aa2f9c2c7d.png">
Figure 3: Overview of our SINet framework, which consists of two main components: the receptive field (RF) and partial decoder component (PDC). The RF is introduced to mimic the structure of RFs in the human visual system. The PDC reproduces the search and identification stages of animal predation. SA = search attention function described in [71]. See x 4 for details.

### How to Run:
> Run Jupyter Notebook in Google colab with GPU.
> run eval_data.m file to calculate quantitative results of generated masks.

### Results:

### Video Results
https://user-images.githubusercontent.com/90741568/234667644-930d9634-283e-4009-9c8a-43344ba16bb6.mp4

https://user-images.githubusercontent.com/90741568/234667742-a6e02a92-8981-44aa-80af-4cabe57f70eb.mp4

https://user-images.githubusercontent.com/90741568/234667769-685f7914-b0bf-4346-a379-3ffff1475f46.mp4

### Image Results 

![download](https://user-images.githubusercontent.com/90741568/234703723-ada7314e-4827-48cc-84e1-dc29b85e095d.png)
![download (2)](https://user-images.githubusercontent.com/90741568/234703754-c3f409bb-642c-44d7-8995-7b19ff93c7fd.png)
![download (1)](https://user-images.githubusercontent.com/90741568/234703762-ecb990ea-c624-4089-85ad-79e0384948bf.png)

### Quantitative Results:

![Screenshot 2023-04-26 171328](https://user-images.githubusercontent.com/90741568/234704232-ca638006-b1af-412b-b9da-32af2067f90b.png)

### References

- Ji, G.-P., Fan, D.-P., Chou, Y.-C., Dai, D., Liniger, A., & Van Gool, L. (2022). Deep Gradient Learning for Efficient Camouflaged Object Detection. Computer Vision and Pattern Recognition. doi:10.48550/ARXIV.2205.12853
- Liu, Y., Wang, C.-q., & Zhou, Y.-j. (2021). Camouflaged people detection based on a semi-supervised search identification network. Defence Technology. doi:https://doi.org/10.1016/j.dt.2021.09.004
- Pei, J., Cheng, T., Fan, D.-P., Tang, H., Chen, C., & Van Gool, L. (2022). OSFormer: One-Stage Camouflaged Instance Segmentation with Transformers. doi:10.48550/ARXIV.2207.02255


