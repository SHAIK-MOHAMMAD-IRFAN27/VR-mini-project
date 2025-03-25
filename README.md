# VR-MINI_PROJECT

-------------------------------------------------------------------------------------------------

##  TASK 2 : WITH/WITHOUT MASK CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS
The tasl is yo classify the images of people with_mask and without_mask.The CNN architecture contains 2 tasks . One is convolution and other is pooling . Convolution is used for FEATURE_EXTRACTION where as pooling is used for spacial reduction and making it a translational_invariant.
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
- **Mean IoU obtained: 0.11**.
- **Experimented with different parameters**, but **results remained low**.
- **Possible reasons for low IoU**:
  - Poor seed selection.
  - High intensity variations.
- **Next Step**: **Switching to GrabCut algorithm for improved segmentation**.

# **2.Region Segmentation Using GrabCut**



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

### **Summary**

- Implemented **GrabCut segmentation**.
- Used **bounding box initialization**.
- Computed **IoU** to evaluate results.
- **Mean IoU obtained: 0.4** (higher than region growing).
- **Improvements over Region Growing**:
  - GrabCut **adapts better** to object boundaries.
  - **Less sensitive** to initial conditions.
  - Works well with **complex textures**.
-Even this is working better than growing region technique but it is getting medium level accuracy , which is not enough for segmentation tasks , so we have to use cnn based algorithms for segmenting.


    
      
      
