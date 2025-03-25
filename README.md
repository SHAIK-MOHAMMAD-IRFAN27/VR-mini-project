# VR-MINI_PROJECT

-------------------------------------------------------------------------------------------------



### DATASET 1:
- The 1st dataset used for task 1 and 2 contains 2 folders . one folder is with_mask and other folder is without_mask .
- with_mask is the folder where the images conatins people with mask and without_mask contains images of people without mask .
- It's a dataset used for  classification task .
### DATASET 2 :
 - The dataset contains face_crop and face_crop segmented folders which are the segmented and corresponding  non-segmented images of a person with mask .
 - The face_crop directory is the X and face_crop_segmentation directory is the target Y for __UNET__  and also the region based segmentation using traditional techniques.
 - It also contains img folder but it doesn't have any corresponding segmented images so we didn't used it .
   


-------------------------------------------------------------------------------------------------
##   TASK 1 : WITH/WITHOUT MASK CLASSIFICATION USING HAND CRAFTED FEATURE EXTRACTION TECHNIQUES.

### INTRODUCTION :
This task is to classify the images of people with_mask and without_mask without using a convolutional neural network . Normally a CNN does feature extraction but here it should be done by us and the give the features extracted to a __SVM classifier__ or a __Multi Layer Perceptron__  to classify the images with_mask and without_mask .


## Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install opencv-python numpy matplotlib seaborn scikit-image scikit-learn
```

## Code Explanation

### 1. Importing Required Libraries
```python
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
```
- `cv2`: Used for image processing
- `numpy`: Handles numerical operations
- `skimage.feature`: Extracts texture-based features
- `sklearn`: Used for training models and evaluating performance

### 2. Defining Directories and Image Size
```python
without_mask_dir = "/content/drive/My Drive/FaceMaskDataset/without_mask"
with_mask_dir = "/content/drive/My Drive/FaceMaskDataset/with_mask"
IMAGE_SIZE = (128, 128)
```
- The dataset contains two folders: `without_mask` and `with_mask`
- Images are resized to **128x128** pixels for consistency

### 3. Feature Extraction Function
```python
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Color Histogram
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 2. Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    # 3. GLCM (Texture Features)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # 4. HOG Features
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), feature_vector=True)

    # Concatenating all features
    feature_vector = np.hstack([hist, lbp_hist, [contrast, correlation], hog_features])
    return feature_vector
```
**Features Extracted:**
- **Color Histogram:** Captures color distribution across RGB channels
- **Local Binary Pattern (LBP):** Captures texture patterns in grayscale images
- **GLCM (Gray Level Co-occurrence Matrix):** Captures contrast and correlation in textures
- **Histogram of Oriented Gradients (HOG):** Extracts edge-based features

### 4. Loading Dataset
```python
features, labels, image_paths = [], [], []
for category, label in [(without_mask_dir, 0), (with_mask_dir, 1)]:
    for file in os.listdir(category):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(category, file)
            try:
                feature_vector = extract_features(img_path)
                features.append(feature_vector)
                labels.append(label)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Skipping {file}: {e}")
```
- Iterates through both directories and extracts features for each image
- Stores features, labels, and image paths for later visualization

### 5. Splitting Dataset
```python
features = np.array(features, dtype=np.float32)
labels = np.array(labels)
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(features, labels, image_paths, test_size=0.2, random_state=42)
```
- **80% for training, 20% for testing**

### 6. Training Models
#### SVM Classifier
```python
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_prob = svm_model.predict_proba(X_test)[:, 1]
```
- Uses **Linear Kernel** for SVM

#### 3-Layer MLP Neural Network
```python
mlp_model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, activation='relu', solver='adam', random_state=42)
mlp_model.fit(X_train, y_train)
mlp_pred = mlp_model.predict(X_test)
mlp_prob = mlp_model.predict_proba(X_test)[:, 1]
```
- Uses **ReLU activation** and **Adam optimizer** and three layers.

### 7. Evaluating Models
```python
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))
print("3-Layer MLP Classification Report:\n", classification_report(y_test, mlp_pred))
```
- Generates precision, recall, and F1-score

### 8. Confusion Matrices
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("SVM Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, mlp_pred), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("3-Layer MLP Confusion Matrix")
plt.show()
```

### 9. ROC Curve Comparison
```python
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_prob)
auc_svm = auc(fpr_svm, tpr_svm)
auc_mlp = auc(fpr_mlp, tpr_mlp)
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.2f})", color="blue")
plt.plot(fpr_mlp, tpr_mlp, label=f"MLP (AUC = {auc_mlp:.2f})", color="green")
plt.legend()
plt.show()
```
The **ROC Curve** is a graphical representation of a classifier’s performance across different threshold values. It plots:  
- **True Positive Rate (TPR) (Sensitivity)** on the Y-axis  
- **False Positive Rate (FPR)** on the X-axis
- Higher AUC → Better Model Performance
## Summary
We used **handcrafted features**, including color histograms, LBP, GLCM, and HOG. These features effectively distinguish masked and unmasked faces. The **3-layer MLP** further improves classification by learning complex patterns in the feature space.

## Conclusion
This project successfully classifies images with and without masks using **SVM** and **MLP** with a high accuracy of **~99%** due to effective feature extraction and model tuning.





-------------------------------------------------------------------------------------------------
##  TASK 2 : WITH/WITHOUT MASK CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS
### INTRODUCTION :
The tasK is to classify the images of people with_mask and without_mask.The CNN architecture contains 2 tasks . One is convolution and other is pooling . Convolution is used for FEATURE_EXTRACTION where as pooling is used for spacial reduction and making it a translational_invariant.
## 1. IMAGE PREPROCESSING :
### STEPS :
  - Mount the google drive
  - Download all the images from github to google drive .
    ```sh
    import os
    import requests
    
    
    repo_owner = "chandrikadeb7"
    repo_name = "Face-Mask-Detection"
    folder_path = "dataset/with_mask"
    
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    
    save_dir = "/content/drive/MyDrive/FaceMaskDataset/with_mask"
    os.makedirs(save_dir, exist_ok=True)
    
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        files = response.json()
    
        for file in files:
            if file["name"].endswith((".jpg", ".jpeg", ".png")):  # Filter images
                img_url = file["download_url"]
                img_name = file["name"]
                img_path = os.path.join(save_dir, img_name)
    
                # Download and save the image
                img_data = requests.get(img_url).content
                with open(img_path, "wb") as handler:
                    handler.write(img_data)
    
                print(f"Downloaded: {img_name}")
    
        print(" All images saved in Google Drive!")
    else:
        print(" Failed to fetch file list. Check the repository URL or API rate limits.")
    

  - Install all the dependencies of libraries and import the libraries .
    ```sh
    !pip install opencv-python tensorflow
    import tensorflow as tf
    import os
    import cv2
    import imghdr
    from google.colab.patches import cv2_imshow
    import numpy as np
    from matplotlib import pyplot as plt
    import numpy as np
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,AveragePooling2D,Dropout, BatchNormalization
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau

  - Load the path directories of dataset from the drive[data_dir] . In that we have 2 classes  which are with_mask_dir and without_mask_dir.
    ```sh
    data_dir="/content/drive/My Drive/FaceMaskDataset"
    without_mask_dir = "/content/drive/My Drive/FaceMaskDataset/without_mask"
    with_mask_dir = "/content/drive/My Drive/FaceMaskDataset/with_mask"
  - We have total 2000 images combined with and without mask.
  - Using cv2 read the images and cross-check.
    ```sh
    image_path=os.path.join(without_mask_dir, without_mask_images[5])
    image = cv2.imread(image_path)
    cv2_imshow(image)
    
  - This converts the images in the paths into a dataset of 2 classes 0 and 1 .
    ```sh
    data=tf.keras.utils.image_dataset_from_directory(data_dir)
    
  - tf.keras.utils.image_dataset_from_directory is a Tensorflow utility function that helps to load images from directory and create a dataset for training deep learning models .
  - By default we have batch_size=32 and image_size=256x256 , color_mode=RGB
    ```sh
    data_iterator=data.as_numpy_iterator()
  - This converts a tensorflow dataset into a numpy iterator which is used to iterate over the dataset and work with numpy arrays .
    ```sh
    batch=data_iterator.next()
  - This creates batch of our dataset .
  - Each batch is a tuple containing (images,labels). image.shape= (32,256,256,3) and labels.shape=(32,)
    ```sh
    data = data.map(lambda x,y: (x/255, y))
  - We have to scale the data of images to between 0 and 1 . so divide the data x by 255.0
  - 
  - We have to split the data into train , test , validation data with a stratification so that imbalance in data doesn't happen .
  - To make this
1. Convert dataset into a numpy array
   ```sh
   X = []
   y = []
   for img_batch, label_batch in data.as_numpy_iterator():
       X.extend(img_batch)
       y.extend(label_batch)
   X = np.array(X)
   y = np.array(y)

2. split the dataset 70%--> train , 20%-->validation and 10% --> test data .
   ```sh
   X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
   X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
   print("Train class distribution:", np.bincount(y_train))
   print("Validation class distribution:", np.bincount(y_val))
   print("Test class distribution:", np.bincount(y_test))
3. Again make the splitted dataset into a tensorflow batches pf size=32
   ```sh
   train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
   val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
   test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
# CREATE A CNN MODEL
## MODEL 1 :
- Activation =Relu
- Optimizer = Adam with initial_lr=0.001
- Regularization = Dropout
- Normalization = BatchNorm
- Batch_size = 32
- Number of epochs trained = 10
- Number of layers = 4 Convolution , 1 dense , 1 output
- Output Activation = Sigmoid
- loss = Binary_crossentropy
- Total number of parameters = 22,576,833
- Metrics = Accuracy , precision , Recall
- Pooling filter_size = (5,5)-->(5,5)-->(3,3)-->(3,3)
- feature_maps = 32-->64-->128-->256
- learning_rate_scheduler = monitor val_loss and if it doesn't chage for 3 epochs decrease the learning rate by a factor of 1/2.
- accuracy: 0.9893 - loss: 0.4554 - precision: 0.9891 - recall: 0.9889 - val_accuracy: 0.8625 - val_loss: 6.9310 - val_precision: 0.7935 - val_recall: 0.9800 - learning_rate: 5.0000e-04
- Test Accuracy: 0.840  ,Test Precision: 0.761 ,Test Recall: 0.9900
## MODEL 2 :
  - Activation = Leaky_Relu
  - Optimizer = SGD with momentum=0.9
  - Regularization = Dropout 0.5 and 0.25
  - Normalization = BatchNorm
  - Batch_size = 16
  - Number of epochs trained = 10
  - Number of layers = 4 Convolution , 1 dense , 1 output
  - Output Activation = Sigmoid
  - loss = Binary_crossentropy
  - Total number of parameters = 22,576,833
  - Metrics = Accuracy , precision , Recall
  - Pooling filter_size = (5,5)-->(5,5)-->(3,3)-->(3,3)
  - feature_maps = 32-->64-->128-->256
  - learning_rate_scheduler = monitor val_loss and if it doesn't chage for 3 epochs decrease the learning rate by a factor of 1/2.
  - accuracy: accuracy: 0.9797 - loss: 0.8187 - precision_1: 0.9818 - recall_1: 0.9766 - val_accuracy: 0.9525 - val_loss: 1.0962 - val_precision_1: 0.9738 - val_recall_1: 0.9300 - learning_rate: 0.0100
  - Test Accuracy: 0.9600 , Test Precision: 0.9792 ,Test Recall: 0.9400
## MODEL 3 :
  - Activation = Relu
  - Optimizer = RMSPROP with initial_lr=0.001
  - Regularization = Dropout 0.5 and 0.25
  - Normalization = BatchNorm
  - Batch_size = 64
  - Number of epochs trained = 10
  - Number of layers = 4 Convolution , 1 dense , 1 output
  - Output Activation = Sigmoid
  - loss = Binary_crossentropy
  - Total number of parameters = 22,576,833
  - Metrics = Accuracy , precision , Recall
  - Pooling filter_size = (5,5)-->(5,5)-->(3,3)-->(3,3)
  - feature_maps = 32-->64-->128-->256
  - learning_rate_scheduler = monitor val_loss and if it doesn't chage for 3 epochs decrease the learning rate by a factor of 1/2.
  - accuracy: 0.9884 - loss: 0.7657 - precision_2: 0.9866 - recall_2: 0.9897 - val_accuracy: 0.9750 - val_loss: 1.8097 - val_precision_2: 0.9657 - val_recall_2: 0.9850 - learning_rate: 5.0000e-04
  - Test Accuracy: 0.9650 , Test Precision: 0.9697 ,Test Recall: 0.9600

## MODEL 4 :
  - Activation = Relu
  - Optimizer = ADAM with initial_lr=0.001
  - Regularization = Dropout 0.5 and 0.25
  - Normalization = BatchNorm
  - Batch_size = 32
  - Number of epochs trained = 10
  - Number of layers = 4 Convolution , 1 dense , 1 output
  - Output Activation = TANH
  - loss = Binary_crossentropy
  - Total number of parameters = 22,576,833
  - Metrics = Accuracy , precision , Recall
  - Pooling filter_size = (5,5)-->(5,5)-->(3,3)-->(3,3)
  - feature_maps = 32-->64-->128-->256
  - learning_rate_scheduler = monitor val_loss and if it doesn't chage for 3 epochs decrease the learning rate by a factor of 1/2.
  - accuracy: 0.9394 - loss: 0.3916 - precision: 0.9323 - recall: 0.9452 - val_accuracy: 0.8850 - val_loss: 0.5162 - val_precision: 0.8208 - val_recall: 0.9850 - learning_rate: 0.0010
  - Test Accuracy: 0.9100 , Test Precision: 0.8475 ,Test Recall: 1.000

## MODEL 5 :
  - Activation = Leaky_Relu
  - Optimizer = ADAGRAD with initial_lr=0.01
  - Regularization = Dropout 0.5 and 0.25
  - Normalization = BatchNorm
  - Batch_size = 16
  - Number of epochs trained = 10
  - Number of layers = 4 Convolution , 1 dense , 1 output
  - Output Activation = SOFTMAX
  - loss = categorical_crossentropy
  - Total number of parameters = 22,576,833
  - Metrics = Accuracy , precision , Recall
  - Pooling filter_size = (5,5)-->(5,5)-->(3,3)-->(3,3)
  - feature_maps = 32-->64-->128-->256
  - learning_rate_scheduler = monitor val_loss and if it doesn't chage for 3 epochs decrease the learning rate by a factor of 1/2.
  - accuracy: 0.9942 - loss: 0.3422 - precision_3: 0.9942 - recall_3: 0.9942 - val_accuracy: 0.9350 - val_loss: 1.0558 - val_precision_3: 0.9350 - val_recall_3: 0.9350 - learning_rate: 0.0050
  - Test Accuracy: 0.9550 , Test Precision: 0.9550 ,Test Recall: 0.9550
## MODEL 6 :
  - Activation = Relu
  - Optimizer = ADADELTA with initial_lr=1.0
  - Regularization = Dropout 0.5 and 0.25
  - Normalization = BatchNorm
  - Batch_size = 64
  - Number of epochs trained = 10
  - Number of layers = 4 Convolution , 1 dense , 1 output
  - Output Activation = SOFTMAX
  - loss = categorical_crossentropy
  - Total number of parameters = 22,576,833
  - Metrics = Accuracy , precision , Recall
  - Pooling filter_size = (5,5)-->(5,5)-->(3,3)-->(3,3)
  - feature_maps = 32-->64-->128-->256
  - learning_rate_scheduler = monitor val_loss and if it doesn't chage for 3 epochs decrease the learning rate by a factor of 1/2.
  - accuracy: 0.9926 - loss: 0.3769 - precision_5: 0.9926 - recall_5: 0.9926 - val_accuracy: 0.9600 - val_loss: 0.9102 - val_precision_5: 0.9600 - val_recall_5: 0.9600 - learning_rate: 0.2500
  - Test Accuracy: 0.9700 , Test Precision: 0.9700 ,Test Recall: 0.9700
-------------------------------------------------------------------------------------------------

- __ReLU__  is commonly used in deep learning due to its simplicity and efficiency in avoiding vanishing gradients.

- __Leaky ReLU__ (Model 2, Model 5) allows small gradients when inputs are negative, which can help in avoiding dead neurons.

- __Tanh__ (Model 4) is less commonly used in CNNs due to saturation issues.

- __Softmax__ (Model 5, Model 6) is used for multi-class classification rather than binary classification.
- __ADAM__ adapts learning rates for each parameter and usually converges faster and does the bias correction.
- __SGD WITH MOMENTUM__ helps in reducing oscillations and is generally better for generalization.
- __RMSprop__  is designed for non-stationary problems and works well in some cases and has initialization issues .
- __Adagrad__  performs well when dealing with sparse data but can reduce learning rate too aggressively.
- __Adadelta__  dynamically adapts learning rates without requiring manual tuning as weights will converge faster .

1. Here __ADADELTA__ did best performance because unlike other models it doesn't need the initialization of learning ratr . Even __ADAM__ needs a initial learning rate . This helps the __ADADELTA__ not to over shoot the optima and reach faster than other models .
2. __ADAM__  models had very high training accuracy (~99%) but much lower test accuracy because its adaptive , it speed ups the convergence but can lead to overfitting with 22Million parameters . And other reason was the __initial learning rate=0.001__ which was very high .
3. __SGD MOMENTUM__ worked well because momentum = 0.9 means each time we are telling it to move in the direction of the gradient with 0.9 confidence means __new_W=old_W + (delta_W+0.9*delta_W )__. But it may have oscillations.
4. __RMSPROP__ also worked well but not as good as __ADADELTA__ because it has the initialization of learning rate issues . 

### CONCLUSION :
__CNN__ is used for Classification tasks without the need of any hand crafted feature_extraction . __Adaptive learning_rate__ works great for most of the tasks . __SoftMax__ is used for __multi-class__ classification and __Sigmoid__ used for __binary__ classification . __Leaky Relu__ is used not to saturate even in the negative values of weights while __Relu__ saturates for negative weights.






-------------------------------------------------------------------------------------------------


## TASK 3 : REGION BASED SEGMENTATION USING TRADITIONAL TECHNIQUES

## **Objective**

### Implement a region-based segmentation method (e.g., thresholding, edge detection) to segment the mask regions for faces identified as "with mask." Visualize and evaluate the segmentation results.

---

## **Implementation Details**
## **1.Region Growing Segmentation**
### **1. Mounting Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

- Google Drive is mounted to access image files stored in the cloud.
- This allows easy retrieval and storage of segmentation results.

### **2. Defining Paths for Input and Output Data**

```python
import os

# Paths
image_folder = "/content/drive/MyDrive/MSFD/1/face_crop/"
mask_gt_folder = "/content/drive/MyDrive/MSFD/1/face_crop_segmentation/"
region_growing_output_folder = "/content/drive/MyDrive/MSFD/1/region_growing_masks/"

os.makedirs(region_growing_output_folder, exist_ok=True)
```

- The **image_folder** stores images of faces.
- The **mask_gt_folder** stores ground truth segmentation masks.
- The **region_growing_output_folder** is created to store the segmentation results.

### **3. Listing Image and Mask Files**

```python
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
mask_files = [f for f in os.listdir(mask_gt_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

num_images_to_process = len(image_files)
print(f"Processing {num_images_to_process} images.")
```

- The script scans the directories and retrieves all image and mask files.
- Only files with extensions `.png`, `.jpg`, or `.jpeg` are considered.

### **4. Region Growing Segmentation**

```python
import cv2
import numpy as np

def region_growing(image, seed, threshold=15):
    """
    Perform region growing segmentation.
    - image: Grayscale input image
    - seed: (x, y) starting point
    - threshold: Maximum intensity difference allowed for region growing
    """
    h, w = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=np.uint8)
    
    stack = [seed]  # Start with the given seed pixel
    initial_intensity = image[seed]  # Get seed intensity

    while stack:
        x, y = stack.pop()
        
        if visited[x, y] == 1:
            continue  # Skip already visited pixels
        
        visited[x, y] = 1  # Mark as visited
        
        # Check if intensity difference is within threshold
        if abs(int(image[x, y]) - int(initial_intensity)) <= threshold:
            segmented[x, y] = 255  # Mark as foreground
            
            # Add 4-neighbor pixels (ensure within image bounds)
            if x > 0: stack.append((x-1, y))
            if x < h-1: stack.append((x+1, y))
            if y > 0: stack.append((x, y-1))
            if y < w-1: stack.append((x, y+1))
    
    return segmented
```

- **Region Growing Algorithm**:
  - Starts from a **seed pixel**.
  - Grows the region based on **intensity similarity**.
  - Expands to neighboring pixels within a **threshold**.
  - Stops when no more pixels meet the criteria.

### **5. Computing IoU and Evaluating Segmentation**

```python
from tqdm import tqdm
from sklearn.metrics import jaccard_score

ious = []
best_iou = 0
best_image = None

num_images = len(image_files)  

for i in tqdm(range(num_images), total=num_images):
    img_name = image_files[i]
    mask_name = mask_files[i]
    
    img_path = os.path.join(image_folder, img_name)
    mask_path = os.path.join(mask_gt_folder, mask_name)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Ground truth mask
    
    if img.shape != mask_gt.shape:
        mask_gt = cv2.resize(mask_gt, (img.shape[1], img.shape[0]))

    mask_gt = (mask_gt > 127).astype(np.uint8)  # Convert to binary
    seed = (img.shape[0]//2, img.shape[1]//2)

    segmented_mask = region_growing(img, seed)

    segmented_binary = (segmented_mask > 127).astype(np.uint8)
    iou = jaccard_score(mask_gt.flatten(), segmented_binary.flatten(), average='binary')
    ious.append(iou)

    if iou > best_iou:
        best_iou = iou
        best_image = (img, segmented_mask, mask_gt, img_name)

    output_path = os.path.join(region_growing_output_folder, img_name)
    cv2.imwrite(output_path, segmented_mask)
```

- Computes **Intersection over Union (IoU)**.
- Identifies the **best-segmented image**.
- Saves the **segmentation output**.

### **6. Displaying Results**

```python
import matplotlib.pyplot as plt

mean_iou = np.mean(ious)
print(f"Mean IoU for Region Growing on images: {mean_iou:.4f}")

if best_image:
    img, segmented_mask, mask_gt, img_name = best_image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_mask, cmap='gray')
    plt.title("Region Growing Segmentation")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask_gt, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.suptitle(f"Best Segmentation - {img_name} (IoU: {best_iou:.4f})")
    plt.show()
```

- Displays the **original image**, **segmented mask**, and **ground truth mask**.
- Shows the **best segmentation result**.

---

## **Summary**

- Implemented **region growing segmentation**.
- Used **seed selection** and **intensity thresholding**.
- Computed **IoU** to evaluate results.
- **Mean IoU obtained: 0.1389**.
- **Experimented with different parameters**, but **results remained low**.
- **Possible reasons for low IoU**:
  - Poor seed selection.
  - High intensity variations.
- **Next Step**: **Switching to GrabCut algorithm for improved segmentation**.

# **2.Region Segmentation Using GrabCut (3 Marks)**



## **Implementation Details**

### **1. Mounting Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

- Google Drive is mounted to access image files stored in the cloud.
- This allows easy retrieval and storage of segmentation results.

### **2. Defining Paths for Input and Output Data**

```python
import os

# Paths
image_folder = "/content/drive/MyDrive/MSFD/1/face_crop/"
mask_gt_folder = "/content/drive/MyDrive/MSFD/1/face_crop_segmentation/"
grabcut_output_folder = "/content/drive/MyDrive/MSFD/1/grabcut_masks/"

os.makedirs(grabcut_output_folder, exist_ok=True)
```

- The **image_folder** stores images of faces.
- The **mask_gt_folder** stores ground truth segmentation masks.
- The **grabcut_output_folder** is created to store the segmentation results.

### **3. Listing Image and Mask Files**

```python
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
mask_files = [f for f in os.listdir(mask_gt_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

num_images_to_process = len(image_files)
print(f"Processing {num_images_to_process} images.")
```

- The script scans the directories and retrieves all image and mask files.
- Only files with extensions `.png`, `.jpg`, or `.jpeg` are considered.

### **4. GrabCut Segmentation**

```python
import cv2
import numpy as np

def apply_grabcut(image):
    """
    Apply GrabCut algorithm for segmentation.
    - image: Input image in BGR format
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    height, width = image.shape[:2]
    rect = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))
    
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask[:, :, np.newaxis]
    
    return segmented_image, mask * 255
```

- **GrabCut Algorithm**:
  - Uses **iterative graph cuts** for foreground extraction.
  - Requires an **initial bounding box** around the object.
  - Separates foreground and background by **energy minimization**.

### **5. Computing IoU and Evaluating Segmentation**

```python
from tqdm import tqdm
from sklearn.metrics import jaccard_score

ious = []
best_iou = 0
best_image = None

num_images = len(image_files)

for i in tqdm(range(num_images), total=num_images):
    img_name = image_files[i]
    mask_name = mask_files[i]
    
    img_path = os.path.join(image_folder, img_name)
    mask_path = os.path.join(mask_gt_folder, mask_name)
    
    img = cv2.imread(img_path)
    mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img.shape[:2] != mask_gt.shape:
        mask_gt = cv2.resize(mask_gt, (img.shape[1], img.shape[0]))
    
    mask_gt = (mask_gt > 127).astype(np.uint8)
    segmented_image, segmented_mask = apply_grabcut(img)
    
    segmented_binary = (segmented_mask > 127).astype(np.uint8)
    iou = jaccard_score(mask_gt.flatten(), segmented_binary.flatten(), average='binary')
    ious.append(iou)
    
    if iou > best_iou:
        best_iou = iou
        best_image = (img, segmented_mask, mask_gt, img_name)
    
    output_path = os.path.join(grabcut_output_folder, img_name)
    cv2.imwrite(output_path, segmented_mask)
```

- Computes **Intersection over Union (IoU)**.
- Identifies the **best-segmented image**.
- Saves the **segmentation output**.

### **6. Displaying Results**

```python
import matplotlib.pyplot as plt

mean_iou = np.mean(ious)
print(f"Mean IoU for GrabCut on images: {mean_iou:.4f}")

if best_image:
    img, segmented_mask, mask_gt, img_name = best_image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_mask, cmap='gray')
    plt.title("GrabCut Segmentation")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask_gt, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.suptitle(f"Best Segmentation - {img_name} (IoU: {best_iou:.4f})")
    plt.show()
```

- Displays the **original image**, **segmented mask**, and **ground truth mask**.
- Shows the **best segmentation result**.

---

## **CONCLUSION**

- Implemented **GrabCut segmentation**.
- Used **bounding box initialization**.
- Computed **IoU** to evaluate results.
- **Mean IoU obtained: 0.4637** (higher than region growing).
- **Improvements over Region Growing**:
  - GrabCut **adapts better** to object boundaries.
  - **Less sensitive** to initial conditions.
  - Works well with **complex textures**.
-Even this is working better than growing region technique but it is getting medium level accuracy , which is not enough for segmentation tasks , so we have to use cnn based algorithms for segmenting.


    
      
      
-------------------------------------------------------------------------------------------------





## TASK 4 :IMAGE SEGMENTATION USING U-NET CNN ARCHITECTURE

### INTRODUCTION :
U-net is a convolutional neural network which is used for image segmetation . It was introduced in 2015 .The name U-net comes from it's __U__ shaped architecture with symmetric encoder-decoder structure. In the encoder side we do the contraction and on the decoder side we do the expansion . We also use the skip connections to the decoder layer from the corresponding encoder layer . In the Unet each pixel is categorized as fore-ground and back-ground .

### DATASET :
 - The dataset contains face_crop and face_crop segmented folders which are the segmented and corresponding  non-segmented images of a person with mask .
 - The face_crop directory is the X and face_crop_segmentation directory is the target Y.
### WORKING :
 - U-Net is a seguence of __conv-conv-pooling__ layers.It has 2 parts . Left side--> contraction , right_side--> expansion paths.
 - In the left side the unet extracts important features . Convolution detects small patterns like edges textures (nose,ears,mouthetc.,.) and pooling reduces the image size to maintain __translational invariance__ .
 - As the image goes deeper , more features will be extraxcted .
 - At the botle neck, U-net has high feature representations . But the model might have forgotten about the borders of the objects etc.,. So, we do the expansion part.
 - During expansion, we try to restore the image borders , but it will be more boxed and pixelled . so , we use the previous memory from the encoder and add a skip connection to the corresponding decoder part to retain the information lost .
 - Now , the edges are deceted sharply .
 - And in the end each pixel is classified as foreground and background . If the pixel is like for suppose mask, then it will be 255 , else its 0.


# CODE :
- Mount the google drive .
  ```sh
  from google.colab import drive
  drive.mount('/content/drive')
- Install and import all the dependencies .
  ```sh
  import tensorflow as tf
  import os
  import cv2
  import imghdr
  from google.colab.patches import cv2_imshow
  import numpy as np
  from matplotlib import pyplot as plt
  import numpy as np
  from sklearn.model_selection import train_test_split
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,AveragePooling2D,Dropout, BatchNormalization
  from tensorflow import keras
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras.regularizers import l2
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import ReduceLROnPlateau

- We have 3 paths . Data_dir , segmented_dir and non_segmented_dir.
  ```sh
  data_dir="/content/drive/My Drive/VR_MINI_C"
  non_segmented_dir = "/content/drive/My Drive/VR_MINI_C/face_crop"
  segmented_dir = "/content/drive/My Drive/VR_MINI_C/face_crop_segmentation"
- Using this code, i have deleted some images from dataset since it was continuously crashing the session using the below code .
  ```sh
  segmented_folder = '/content/drive/My Drive/VR_MINI_C/face_crop_segmentation'
  non_segmented_folder = '/content/drive/My Drive/VR_MINI_C/face_crop'

  temp_segmented_images = set(os.listdir(segmented_folder))
  temp_non_segmented_images = set(os.listdir(non_segmented_folder))
  
  
  common_images = temp_segmented_images.intersection(temp_non_segmented_images)
  
  for file in temp_segmented_images:
      if file not in common_images:
          file_path = os.path.join(segmented_folder, file)
          os.remove(file_path)
          print(f'Removed: {file_path}')
  
  for file in temp_non_segmented_images:
      if file not in common_images:
          file_path = os.path.join(non_segmented_folder, file)
          os.remove(file_path)
          print(f'Removed: {file_path}')
  
  print("Finished removing unmatched files.")

- Changed my images to 128x128 pixels.
  ```sh
  IMG_WIDTH = 128
  IMG_HEIGHT = 128
  IMG_CHANNELS = 3
- In order to load images to the model, we have to convert it to the np.array form . We do it by using below code and save them to segmented_images and non_segmented_images.
  ```sh
  def load_images(image_dir, img_size=(128, 128), save_path=None):
    images = []
    for img_name in os.listdir(image_dir):
      img_path = os.path.join(image_dir, img_name)
      img = cv2.imread(img_path)
      img = cv2.resize(img, img_size)  # Resize to target size
      images.append(img)
    images_array = np.array(images)
    # If a save path is provided, save the images array
    if save_path:
        np.save(save_path, images_array)  # Save as .npy file
        print(f"Images saved to {save_path}")
    return images_array
  save_path1 = '/content/drive/MyDrive/non_segmented_images.npy'
  save_path2 = '/content/drive/MyDrive/segmented_images.npy'
  non_segmented_images = load_images(non_segmented_dir,save_path=save_path1)
  segmented_images = load_images(segmented_dir,save_path=save_path2)   
- Now scale the pixels 0--1 by dividing with 255.0
  ```sh
  segmented_images = segmented_images / 255.0
  non_segmented_images = non_segmented_images / 255.0
- As the output should by in the 128x128x1 dimensions , but the face_crop_segmentation has 3 cannels . so, do the average of 3 RGB channels .
  ```sh
  segmented_images = np.mean(segmented_images, axis=-1,keepdims=True) 
- Now , do the dataset splitting . Training data --> 70% validation-->30%
  ```sh
  from sklearn.model_selection import train_test_split
  X_train, X_temp, y_train, y_temp = train_test_split(non_segmented_images, segmented_images, test_size=0.2, random_state=42)
- Now build the U-net model.
  ```sh
  import tensorflow as tf
  
  inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  
  c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
  c1 = tf.keras.layers.Dropout(0.1)(c1)
  c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
  p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
  
  c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
  c2 = tf.keras.layers.Dropout(0.1)(c2)
  c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
  p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
  
  c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
  c3 = tf.keras.layers.Dropout(0.2)(c3)
  c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
  p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
  
  c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
  c4 = tf.keras.layers.Dropout(0.2)(c4)
  c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
  p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
  
  c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
  c5 = tf.keras.layers.Dropout(0.3)(c5)
  c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
  
  u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
  u6 = tf.keras.layers.concatenate([u6, c4])
  c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
  c6 = tf.keras.layers.Dropout(0.2)(c6)
  c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
  
  u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
  u7 = tf.keras.layers.concatenate([u7, c3])
  c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
  c7 = tf.keras.layers.Dropout(0.2)(c7)
  c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
  
  u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
  u8 = tf.keras.layers.concatenate([u8, c2])
  c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
  c8 = tf.keras.layers.Dropout(0.1)(c8)
  c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
  
  u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
  u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
  c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
  c9 = tf.keras.layers.Dropout(0.1)(c9)
  c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
  
  outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
  
  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=['accuracy',iou_metric])
  model.summary()

- Check the shape of inputs and outputs .
  ```sh
  print(type(X_train), X_train.shape)
  print(type(y_train), y_train.shape)

- Now , fit the X_train and y_train .
  ```sh
  history = model.fit(
    X_train, 
    y_train,  
    batch_size=32,
    epochs=15,
    validation_data=(X_temp,y_temp),  
)

- I've used Intersection_over_union as my Mertic along with accuracy and loss as dice_loss
- As we go through in process , the accuracy decreases or stays as it is since , here each pixel is being classified as foreground and background not the image which is commonly done in CNN's.
  ```sh
  def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))
  def iou_metric(y_true, y_pred):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    return intersection / (union + 1e-6)
- I've run the code for 15 epochs .
- Epoch 15/15 113/113 ━━━━━━━━━━━━━━━━━━━━ 623s 5s/step - accuracy: 0.5685 - iou_metric: 0.7023 - loss: 0.1751 - val_accuracy: 0.5736 - val_iou_metric: 0.6986 - val_loss: 0.1776
- IOU for the validation data is nearly (~70%) .
- Normally we say >85% is the best accuracy and used in medical images segmentation .
- 70-85 % is generally we get accuracy for U net .
- After this running we get probabilities for each pixel here . If wach pixel is >0.5 the its 255 , else it's 0.

### CONCLUSION :
By doing this i've learned about the architecture of Unet and got to know how the segmentation is done at each step and each layers and why the skip connections are used and needed for this task . And also got to know why __accuracy__ and  __binary cross entropy__ are the bad metrics and loss functions for U net ,while __IOU score__ and __dice score__  work good .
  
  


