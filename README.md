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


    
    
      
      
