# FDA  Submission

**Your Name:**
Gennady Lungu

**Name of your Device:**
X-Ray Pneumonia Detection Assistant (XR-PDA)

## Algorithm Description 

### 1. General Information

**Intended Use Statement:** 
For assisting a radiologist in detection of pneumonia in x-ray images.

**Indications for Use:**
Screening x-ray images.
Both men and women of ages 2 to 90.  
Restricted to x-ray images with the following properties: 
* Body part: Chest
* Position: AP (Anterior/Posterior) or PA (Posterior/Anterior)
* Modality: DX (Digital Radiography) 
 

**Device Limitations:**
* The model is recommended for use without following comorbid thoracic pathologies:
    * Consolidation
    * Edema
    * Effusion
    * Hernia


**Clinical Impact of Performance:**

Inference can be performed on a common-purpose CPU found on desktop PCs.
Performance has been tested on Intel(R) Xeon(R) @ 2.30GHz CPU. 
Average inference time for a single image: ms 

### 2. Algorithm Design and Function

<< Insert Algorithm Flowchart >>

**DICOM Checking Steps**

The algorithm performs the following checks on the DICOM image:

* Check Patient Age is between 2 and 90 (inclusive)
* Check Examined Body Part is 'CHEST'
* Check Patient Position is either 'PA' (Posterior/Anterior) or 'AP' (Anterior/Posterior)
* Check Modality is 'DX' (Digital Radiography)

**Preprocessing Steps**

The algorithm performs the following preprocessing steps on an image data:

* Converts RGB to Grayscale (if needed)
* Re-sizes the image to 244 x 244 (as required by the CNN)
* Normalizes the intensity to be between 0 and 1 (from original range of 0 to 255)   

**CNN Architecture**

The algorithm uses pre-trained VGG16 Neural Network (except the last block of 
Convolution + Pooling layers that was re-trained), 
with additional 4 blocks of 'Fully Connected + Dropout' layers.

The network output is a single probability value for binary classification.


### 3. Algorithm Training

**Parameters:**
* Types of augmentation used during training:
    * horizontal flip
    * height shift: 0.1
    * width shift: 0.1
    * rotation angle range: 0 to 20 degrees
    * shear: 0.1
    * zoom: 0.1
* Batch size: 32
* Optimizer learning rate: 1e-5
* Layers of pre-existing architecture that were frozen
    * All except last convolution + pooling block 
* Layers of pre-existing architecture that were fine-tuned
    * The last 2 layers of VGG16 network: block5_conv3 + block5_pool
* Layers added to pre-existing architecture
    * flatten_1     (Flatten)         
    * dropout_1     (Dropout)         
    * dense_1       (Dense,   1024)   
    * dropout_2     (Dropout, 0.2)          
    * dense_2       (Dense,   512)    
    * dropout_3     (Dropout, 0.2)
    * dense_3       (Dense,   256)    
    * dropout_4     (Dropout, 0.2)
    * dense_4       (Dense)   1    

#### Algorithm training performance visualization

![Training performance](training_loss_accuracy.png)

The early stopping with patience of 10 epochs was used to stop the learning process.

The behaviour of the validation loss during training may indicate a possibility that
a lower learning rate could produce better results. This is left for future research. 
 

#### Model performance metrics depending on threshold

![ROC Curve](roc_curve.png)

ROC Curve is not particularly impressive, but does show that the model 
has indeed learned something from the data.

![Model Performance by threshold](f1_npv_etc.png)

As we can see, the model has low precision, but higher recall, 
and maintains high negative predictive value. 


**Final Threshold and Explanation:**

The maximum F1 score for the model is 0.404 and it is achieved with threshold value of 0.45..
Comparing this value with those given in 
[CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rayswith Deep Learning](
https://arxiv.org/pdf/1711.05225.pdf):

| Person or Device | F1    | Sensitivity | Specificity | 
|------------------|-------|-------------|-------------|
| Radiologist 1    | 0.383 | 0.309       | 0.453   |
| Radiologist 2    | 0.356 | 0.282       | 0.428   |
| Radiologist 3    | 0.365 | 0.291       | 0.435   |
| Radiologist 4    | 0.442 | 0.390       | 0.492   |
| Radiologist Avg. | 0.387 | 0.330       | 0.442   |
| CheXNet          | 0.435 | 0.387       | 0.481   |
| XR-PDA Max F1    | 0.408 | 0.290       | 0.578   |
 
As we can see, this model achieves higher maximum F1 score than an average radiologist 
in the study. State of the art neural network from the paper achieves higher F1 score, 
but not significantly higher. We will be comparing this model with the 
performance of human radiologists.

Furthermore, since the model does not have a high precision with any meaningful recall value,
its usefulness tends to be in its recall (and negative predictive value).
Therefore, it makes sense to maximize recall and NPV even at the cost of small loss in precision.
A good threshold value that achieves that is 0.377:

| Person or Device | F1    | Sensitivity | Specificity | 
|------------------|-------|-------------|-------------|
| Radiologist 1    | 0.383 | 0.309       | 0.453       |
| Radiologist 2    | 0.356 | 0.282       | 0.428       |
| Radiologist 3    | 0.365 | 0.291       | 0.435       |
| Radiologist 4    | 0.442 | 0.390       | 0.492       |
| Radiologist Avg. | 0.387 | 0.330       | 0.442       |
| XR-PDA Max F1    | 0.408 | 0.290       | 0.578       |
| XR-PDA T=0.377   | 0.404 | 0.268       | 0.441       |


### 4. Databases

**Description of Training Dataset:**

Training dataset consisted of 2290 chest xray images, with a 50/50 split between 
positive and negative cases. 

Example images: 

![Training Set Examples (with Augmentation)](training_aug.png)

**Description of Validation Dataset:** 

Validation dataset consisted of 1430 chest xray images, with 20/80 split between 
positive and negative cases, which more reflects the occurence of pneumonia 
in real world images.
   

### 5. Ground Truth

The data was coming from a larger xray dataset, with disease labels 
that were created using Natural Language Processing (NLP) to mine the associated 
radiological reports. The labels include 14 common thoracic pathologies 
(Pneumonia being one of them): 
- Atelectasis 
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia 

The biggest limitation of this dataset is that image labels were NLP-extracted so there 
could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.

The original radiology reports are not publicly available but more details 
on the labeling process can be found [here](https://arxiv.org/abs/1705.02315). 


### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

The following population subset is to be used for the FDA Validation Dataset:
* Both men and women 
* Age 2 to 90 
* Without known comorbid lung conditions listed above

**Ground Truth Acquisition Methodology:**

Ground truth for the FDA Validation Dataset should be obtained a from practicing radiologist.  

**Algorithm Performance Standard:**

The algorithm can be used on the regular PC with general purpose CPU.
On the Intel(R) Xeon(R) @ 2.30GHz CPU machine performance is as follows:

* Average image pre-processing time: 29 ms, max: 111 ms
* Average inference time: 615 milliseconds, max: 734 ms

So, total inference time does not exceed 850 milliseconds on Intel Xeon CPU. 
The same or similar general-purpose CPU is recommended.