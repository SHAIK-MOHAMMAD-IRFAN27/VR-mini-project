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
    
