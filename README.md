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
    
        print("✅ All images saved in Google Drive!")
    else:
        print("❌ Failed to fetch file list. Check the repository URL or API rate limits.")
    

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
    
    
    
