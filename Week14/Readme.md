
## 1. Dataset Drive Link

(fg, bg, fg_masks in folder, fg_bg, fg_bg_mask, depth images zipped in the folder)
<br>
https://drive.google.com/open?id=1aXOUCyBZn8fL2mL037g7TvYvuTsj7rfg

## 2. Dataset Statistics

| Data | File/Folder Name | Type |Count | Image_size | Datafolder_size | Mean | std |
|---|---|---|---|---|---|---|---|
| Background | bg | Jpg | 102 | 224x224x3 | 1.02 MB |  |   |
| Foreground | fg | png | 104 | Varying size from 90x90x4 to 160x160x4 | 2.25 MB |  |   |
| Foreground Mask | Cars_Mask | png | 104 | Varying size from 90x90x4 to 160x160x4 | 102 KB |  |   |
| Foreground over Background | fg_bg.zip | Jpg | 424320 | 224x224x3 | 7 GB | [0.3931, 0.3785, 0.3606] | [0.1965, 0.1813, 0.1779] |
| Foreground over Background Mask | fg_bg_mask.zip | Jpg | 424320 | 224x224x1 | 2 GB | [0.1630] | [0.3598] |
| Depth Output | depth_output.zip | Jpg | 424320 | 224x224x1 | 764 MB | [0.0878] | [0.0157] |


## 3. Dataset



Background (bg)  Count : 102:
![bg](https://github.com/deepakgowtham/EVA4/blob/master/Week14/Images/bg.png)



Foreground (fg) Count : 104 :
![fg](https://github.com/deepakgowtham/EVA4/blob/master/Week14/Images/fg.png)



Foreground Mask(fg_mask) Count : 104 :
![fg_mask](https://github.com/deepakgowtham/EVA4/blob/master/Week14/Images/fg_mask.png)



Foreground overlayed randomly on Background (fg_bg) Count : 424320 :
![fg_bg](https://github.com/deepakgowtham/EVA4/blob/master/Week14/Images/fg_bg.png)



Mask with same position as fg_bg (fg_bg_mask) Count : 424320 :
![fg_bg_mask](https://github.com/deepakgowtham/EVA4/blob/master/Week14/Images/fg_bg_mask.png)



Depth Map generated (depth) Count : 424320 :
![depth](https://github.com/deepakgowtham/EVA4/blob/master/Week14/Images/depth.png)


## 4. Dataset creation Steps
 
#### 4.1 how were fg created with transparency
fg transparency was created using Gimp. Magic wand tool was used to select the fg and rest were deleted

#### 4.2 how were masks created for fgs
Masks were created using OpenCV library. For the FGs where the pixel value was transparent, it was marked black and wherever pixel value was >=1 it was assigned white in the mask image. <br>Code: https://github.com/deepakgowtham/EVA4/blob/master/Week14/Session15A_Mask_RCNN_CV2.ipynb

#### 4.3 how did you overlay the fg over bg and created 20 variants
For fg_bg : used openCV again to overlay the fg over bg. For every combination of fg and bg, position of the fg was randomly generated 20 times. for every random position generation, 2 fg_bg images were created, one with the fg and other with the fg LR flipped.<br> Image overlay was done using calculation of alpha value of the foreground to maintain transparency of fg over bg.<br> Code: https://github.com/deepakgowtham/EVA4/blob/master/Week14/Session15A_Mask_RCNN_CV2.ipynb

#### 4.4 how did you create your depth images? 
Depth images were created using the DenseDepth model using kitti weights. <br>Source : https://github.com/ialhashim/DenseDepth.git <br> the images were stored as one channel only.<br> Code: https://github.com/deepakgowtham/EVA4/blob/master/Week14/Session15A_Mask_RCNN_Depth.ipynb



## 5. Notebooks used for creation of Dataset.
https://github.com/deepakgowtham/EVA4/blob/master/Week14/Session15A_Mask_RCNN_CV2.ipynb
https://github.com/deepakgowtham/EVA4/blob/master/Week14/Session15A_Mask_RCNN_Depth.ipynb


## 6. Notebook for calculation of Statistics.

https://github.com/deepakgowtham/EVA4/blob/master/Week14/Session15A_Mask_RCNN_stat.ipynb
