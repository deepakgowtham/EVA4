# Assignment 15

### Monocular depth estimation and Mask Generation.

### Dataset Creation.
- Dataset is generated synthetically by placing transparent foreground images over set of backupground images.
- Detailed procedure for creating the dataset is mentioned [here](https://github.com/deepakgowtham/EVA4/blob/master/Week14/Readme.md)
- Dataset download [link](https://drive.google.com/open?id=1aXOUCyBZn8fL2mL037g7TvYvuTsj7rfg)

### Approach 
- [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Hu_Visualization_of_Convolutional_Neural_Networks_for_Monocular_Depth_Estimation_ICCV_2019_paper.pdf) Junjie Hu, Yan Zhang, Takayuki Okatani, "Visualization of Convolutional Neural Networks for Monocular Depth Estimation," ICCV, 2019 was used source
- [Code](https://github.com/JunjH/Visualizing-CNNs-for-monocular-depth-estimation) Corresponding to the same paper was used and customized.

### Model Details
- There are two models used in this approach
- First a pre-trained resnet model to predict the Mask
- Auto Encoder Decoder to predict the Depth.
- The second model takes input product of the mask and input image.
- The model architecture from the papter is shown below.
![Model Architecture](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/fig_arch.png)

#### Resnet50 Params:

Total params: 67,569,473 <br>
Trainable params: 67,569,473 <br>
Non-trainable params: 0 <br>
#### Auto Encoder Decoder Params:

Total params: 16,393,752 <br>
Trainable params: 16,393,752 <br>
Non-trainable params: 0 <br>

The full model summary of both the models can be found in this notebook [here](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Notebooks/EVA4_Session15_Model_Params.ipynb)


### Model Training:
For model training the 420k images were split into train(80%), test(20%) images.
The input size of the fg_bg image and the depth images were increased incrementally for in batches for epochs.

#### Epoch 1-2:
- Input Image size: 38x38x3
- Batch size: 512
- Training Time: 1.6 hrs per epoch

##### Fg_Bg Images:
![fg_bg](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch2/fg_bg.png)
##### Target Mask Images:
![Target Mask](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch2/input_mask.png)
##### Target Depth Images:
![Target Depth](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch2/depth_input.png)
##### Predicted Mask Images:
![Predicted Mask](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch2/output_mask.png)
##### Predicted Depth Images:
![Predicted Depth](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch2/output_depth.png)

#### Epoch 3-4:
- Input Image size: 38x38x3
- Batch size: 256
- Training Time: 1 hrs per epoch

##### Predicted Mask Images:
![Predicted Mask](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/epoch4/output_mask.png)
##### Predicted Depth Images:
![Predicted Depth](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/epoch4/output_depth.png)

- The notebook for this epoch can be found [here](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Notebooks/EVA4_Session15_Epoch_4.ipynb)

#### Epoch 5-6:
- Input Image size: 48x48x3
- Batch size: 256
- Training Time: 1 hrs per epoch

##### Predicted Mask Images:
![Predicted Mask](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch6/output_mask.png)
##### Predicted Depth Images:
![Predicted Depth](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch6/output_depth.png)

- The notebook for this epoch can be found [here](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Notebooks/EVA4_Session15_Epoch_6.ipynb)

#### Epoch 6-10:
- Input Image size: 48x48x3
- Batch size: 256
- Training Time: 1 hrs per epoch

##### Predicted Mask Images:
![Predicted Mask](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch10/output_mask.png)
##### Predicted Depth Images:
![Predicted Depth](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch10/output_depth.png)

- The notebook for this epoch can be found [here](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Notebooks/EVA4_Session15_Epoch_10.ipynb)

#### Epoch 10-12:
- Input Image size: 52x52x3
- Batch size: 256
- Training Time: 1 hr 20 Mins per epoch

##### Predicted Mask Images:
![Predicted Mask](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch12/output_mask.png)
##### Predicted Depth Images:
![Predicted Depth](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch12/output_depth.png)

- The notebook for this epoch can be found [here](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Notebooks/EVA4_Session15_Epoch_12.ipynb)

#### Epoch 12-14:
- Input Image size: 64x64x3
- Batch size: 128
- Training Time: 1 hr 40 Mins per epoch

##### Predicted Mask Images:
![Predicted Mask](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch14/output_mask.png)
##### Predicted Depth Images:
![Predicted Depth](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch14/output_depth.png)
![Predicted Depth](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Epoch14/output_depth_2.png)

- The notebook for this epoch can be found [here](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Notebooks/EVA4_Session15_Epoch_14.ipynb)












