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
- For model training the 420k images were split into train(80%), test(20%) images. <br>
- The **input size** of the fg_bg image and the depth images were **increased incrementally** for in batches for epochs.
- The Loss function used for training the model is Cosinesimilarity.


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
- Input Image size: 56x56x3
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


- The notebook for this epoch can be found [here](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Notebooks/EVA4_Session15_Epoch_14.ipynb)

#####  Learning Curve:
![Learning Curve](https://github.com/deepakgowtham/EVA4/blob/master/Week15/Images/Learning_Curve.png)


####  Training data:
- The total training time for all the 14 epochs: 17 hours 40 minutes
- The training loss and test loss reduces when training with the same image size but increases as the training input image size is increased.


#### Training Log

|epoch |MSE | RMSE | ABS_REL | LG10 | MAE | DELTA1 | DELTA2 | DELTA3 
|---|---|---|---|---|---|---|---|---|
|epoch 0 | 0.005257841530445609| 0.07154805335703658, |0.03378925721693452| nan, | 0.041959120314652563, | 0.9884973586414734, | 0.9961986858201063, | 0.9986032799538085|
|epoch 1 | 0.0033398533853116483,| 0.057722808301538725, | 0.02663627428846886, | nan, | 0.033235909959489586, | 0.9918754773262219, | 0.9976736100370888, | 0.9991964970239148|
|epoch 2 | 0.0028815822549224696,| 0.05242482119072706, | 0.022063103171562716, | 0.009295359116572982, | 0.031660318898590535, | 0.9960979843121668, | 0.998667465795399, | 0.9995556819852481|
|epoch 3 | 0.0016247311532026829,| 0.04024092670457036, | 0.017449388319842777, | 0.007449132186286208, | 0.025375543173464354, | 0.9979052429465327, | 0.9993992933320783, | 0.9998188947480549|
|epoch 4 | 0.001978461982434416,| 0.0436104065759689, | 0.01897431124032645, | 0.008075690744683528, | 0.027515347243524065, | 0.9974572418502013, | 0.9992387327583307, | 0.999755818559753|
|epoch 5 | 0.0013422581198102715,| 0.03658528927696776, | 0.01597044753124249, | 0.0068351218647563754, | 0.023308631597398634, | 0.9983855912048892, | 0.9995326513528464, | 0.999863000881618|
|epoch 8 | 0.0015575952713709987,| 0.03903022585197901, | 0.017122361576535808, | 0.007317344185743105, | 0.024942482624983987, | 0.9980812844557639, | 0.9994746803068647, | 0.9998442919905189|
|epoch 9 | 0.0011503646629325186,| 0.03386737901694949, | 0.014693243384114037, | 0.006294656996889149, | 0.021468215346684732, | 0.9986057699086259, | 0.999571843789174, | 0.9999009953365067|
|epoch 10 | 0.0037241011760007943,| 0.06048789370361137, | 0.02676215664052361, | 0.011320678419595076, | 0.03902629736900824, | 0.9963879257575419, | 0.99869644691322, | 0.9994474729441589|
|epoch 11 | 0.0026119751215710123,| 0.05101700555280323, | 0.021937274042556278, | 0.00929900122452443, | 0.031997289737548854, | 0.9972797751336795, | 0.9990007353224546, | 0.999560197868678|
|epoch 12  | 0.00350853538084115,| 0.058663881878294105, | 0.02720961381988895, | nan, | 0.03969067200698059, | 0.9970343634373822, | 0.9989217237469658, | 0.9995362639966594|
|epoch 13 | 0.002626794558418788,| 0.051105982727942874, | 0.023230139093774162, | nan, | 0.03393625332461178, | 0.9976833452825992, | 0.9991162502927478, | 0.9995905529499773|


#### The complete training log all the epochs is recorded as csv file [here](https://drive.google.com/open?id=11eH852GZOBjeBDEP0mqTXoUxWTH5q8Ks)
#### The trained model weights can be found [here](https://drive.google.com/open?id=1fqN14GGBqTNv_ANo_KUp5yJD74xqjcri)
### The complete code can be [here](https://github.com/deepakgowtham/EVA4/tree/master/Week15/Dense_Depth)



## Challanges and Lessons:
- Handling the large data was a challange, the training time per epoch was high which made it mandatory to save the progress of the training everytime directly in google drive instead on colab. Lost few epochs of training because of not saving it in drive.
- Modularising helped us to avoid run time restarts every time of colab as we are running the code as python command.
- Tried using HD5 file system for data loading, the size of stored files increased but loading time reduced with higher batch size but was not able to complete it as I faced issue for loading both the images saved with same name, this was not a problem when loading using csv file as it contained the image name along with the path.
- The model is overfitting. Implementing cut out strategy will help reduce the overfitting.
- Need to implement data augumentation strategies.
- Changed the optimizer from Adam to SGD but it didnt improve both the training time and improve performance.

## References:
- Depth Estimation Basics [link](https://towardsdatascience.com/depth-estimation-1-basics-and-intuition-86f2c9538cd1)
- Depth Estimation Basics2 [link](https://medium.com/beyondminds/depth-estimation-cad24b0099f)
- Storing image in different formats [link](https://realpython.com/storing-images-in-python/)






