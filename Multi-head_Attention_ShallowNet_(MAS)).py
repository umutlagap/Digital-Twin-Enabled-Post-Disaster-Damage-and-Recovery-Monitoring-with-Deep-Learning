# Multi-head Attention ShallowNet (MA_ShallowNet)
# ---------------------------------------------------------------------------------
# Article:Digital Twin-Enabled Post-Disaster Damage and Recovery Monitoring with Deep Learning: Leveraging Transfer Learning, Attention Mechanisms, and Explainable AI
# Multi-head Attention ShallowNet Training Example
#
# Authors: Umut Lagap, Saman Ghaffarian
#
# Description:
# This script provides an example implementation of training a Multi-head Attention 
# RegNet model for post-disaster damage and recovery monitoring using deep learning.
# Users can easily adapt this code to experiment with other attention mechanisms 
# (e.g., Spatial Attention) or omit attention modules entirely to train a baseline model.
# Additionally, the script is compatible with other model architectures, such as 
# InceptionV3 or DenseNet. Users are encouraged to create their own variations 
# or fine-tune the models for their specific datasets and use cases.
#
# Key Features:
# - Implements Multi-head Attention for ShallowNet
# - Preprocessing and augmentation pipelines
# - Training and evaluation procedures
# - Reliability check with Grad-CAM and Saliency Maps
#
# Please refer to the accompanying documentation and article for further details.


# Python version: 3.10.16 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 16:19:12) [MSC v.1929 64 bit (AMD64)]
# NumPy version: 2.2.1
# OpenCV (cv2) version: 4.10.0
# TensorFlow version: 2.18.0
# Keras version (via TensorFlow): 2.18.0
# SciPy version: 1.15.1

# Folder structure of the dataset:
# Train/
# ├── Class1/
# │   ├── image1.jpg
# │   ├── image2.jpg
# │   ├── image3.jpg
# │   └── ...
# ├── Class2/
# │   ├── image1.jpg
# │   ├── image2.jpg
# │   ├── image3.jpg
# │   └── ...
# └── ...

# --------------------------------------------------------------------------
# 0. Import Necessary Libraries
# --------------------------------------------------------------------------

import os
import random
import numpy as np
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add, LayerNormalization, MultiHeadAttention,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------------------
# 1. Reproducibility: Set random seeds
# --------------------------------------------------------------------------
seed = 1453
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# --------------------------------------------------------------------------
# 2. Directory & Hyperparameters
# --------------------------------------------------------------------------
#dir_path = "..."  # enter your dataset folder
# save_directory = "..." #enter your directory path
os.makedirs(save_directory, exist_ok=True)

class_labels = {
    "not_damaged": 0,
    "not_recovered": 1,
    "recovered": 2
}
num_classes = 3  # e.g., for "not_damaged", "not_recovered", "recovered"

target_size = (224, 224)
batch_size = 32
epochs_to_train = 200

# --------------------------------------------------------------------------
# 3. Load All Images + Labels from Single Directory
# --------------------------------------------------------------------------

image_paths = sorted(list(paths.list_images(dir_path)))
X = []
Y = []

for path in image_paths:
    # Example: .../train/<class_name>/some_image.jpg
    label_name = path.split(os.path.sep)[-2]  # get the folder name
    label = class_labels[label_name]

    img = cv2.imread(path)
    if img is None:
        # If OpenCV fails to read the file, skip it or handle error
        print(f"[WARNING] Could not read {path}. Skipping...")
        continue
    
    # Resize to target size
    img = cv2.resize(img, target_size)
    
    X.append(img)
    Y.append(label)

# Convert to NumPy arrays
X = np.array(X, dtype="float32")
Y = np.array(Y, dtype="int")

print(f"Total images loaded: {len(X)}")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# --------------------------------------------------------------------------
# 4. Train/Val/Test Split: 60% / 20% / 20%
# --------------------------------------------------------------------------
# First split off 20% for test => (80% remain)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=seed,
    stratify=Y
)

# Now split the remaining 80% into 60% (train) and 20% (val)
# 20% of 80% is 0.25 => 0.25 * 80% = 20% of the entire dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.25,
    random_state=seed,
    stratify=y_trainval
)

print("[INFO] Split summary:")
print(f"Train size: {X_train.shape[0]} ({X_train.shape})")
print(f"Val   size: {X_val.shape[0]}   ({X_val.shape})")
print(f"Test  size: {X_test.shape[0]}  ({X_test.shape})")

# --------------------------------------------------------------------------
# 5. Data Generators from NumPy Arrays
# --------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Convert labels to one-hot vectors
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat   = to_categorical(y_val,   num_classes=num_classes)
y_test_cat  = to_categorical(y_test,  num_classes=num_classes)

train_generator = train_datagen.flow(
    X_train, y_train_cat,
    batch_size=batch_size,
    shuffle=True,
    seed=seed
)

val_generator = val_datagen.flow(
    X_val, y_val_cat,
    batch_size=batch_size,
    shuffle=False
)

test_generator = test_datagen.flow(
    X_test, y_test_cat,
    batch_size=batch_size,
    shuffle=False
)


# --------------------------------------------------------------------------
# 6. Define a Multi-Head Attention Module
# --------------------------------------------------------------------------
def multi_head_attention_module(input_tensor, num_heads=3, key_dim=32):
    # Multi-Head Attention layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(input_tensor, input_tensor)
    
    # Add & Normalize
    attention_output = Add()([input_tensor, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    return attention_output

# Instead of Multi-head Attention module, Spatial Attention model can be used.
# Spatial Attention
# def spatial_attention(input_tensor):
#     avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
#     max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    
#     concat = Concatenate(axis=-1)([avg_pool, max_pool])
#     spatial_attention_feature = Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    
#     return multiply([input_tensor, spatial_attention_feature])

# --------------------------------------------------------------------------
# 7. Build the Model (MA_ShallowNet)
# --------------------------------------------------------------------------
# Instead of MA_ShallowNet, other models, such as InceptionV3, can be used, or you can create your model.
def shallowNet(input_shape=(224, 224, 3), num_classes=3):
    """
    ShallowNet architecture with Multi-Head Attention for classification.
    
    Args:
        input_shape: Tuple indicating input dimensions (height, width, channels).
        num_classes: Integer representing the number of output classes.
        
    Returns:
        A compiled tf.keras.Model instance.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Convolutional layers
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Apply Multi-Head Attention
    x = multi_head_attention_module(x, num_heads=4, key_dim=32)

    # Flatten and Dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Output layer for classification
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ShallowNet_MHA")

    return model

# Build the model using ShallowNet
model = shallowNet(input_shape=(224, 224, 3), num_classes=num_classes)

# Print the model summary
model.summary()

# --------------------------------------------------------------------------
# 8. Compile the Model
# --------------------------------------------------------------------------
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-4),
    metrics=[
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        'accuracy'
    ]
)

# --------------------------------------------------------------------------
# 9. Callbacks (Checkpoint & Early Stopping)
# --------------------------------------------------------------------------
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(
        save_directory,
        "..." # Name the model
    ),
    save_weights_only=False,
    save_best_only=False,  
    monitor="val_loss",
    mode="min",
    verbose=1
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=20,
    mode="min",
    verbose=1
)

# --------------------------------------------------------------------------
# 10. Train on 60% (train_generator), Validate on 20% (val_generator)
# --------------------------------------------------------------------------
history = model.fit(
    train_generator,
    epochs=epochs_to_train,
    validation_data=val_generator,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# --------------------------------------------------------------------------
# 11. Evaluate on Test
# --------------------------------------------------------------------------
print("\n[INFO] Evaluating on Test Set")
test_loss, test_precision, test_recall, test_acc = model.evaluate(
    test_generator,
    verbose=1
)

print("\n=== Final Test Metrics ===")
print(f"Loss:      {test_loss:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"Accuracy:  {test_acc:.4f}")

# --------------------------------------------------------------------------
# 12. Training and Validation Accuracy/Loss Graphs
# --------------------------------------------------------------------------
plt.figure(figsize=(20,5))

#------------ Accuracy Graph -------
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend(loc="lower right")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy", fontsize=16)

#------------ Loss Graph -------
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend(loc="upper right")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.ylim([0, max(plt.ylim())])
plt.title("Training and Validation Loss", fontsize=16)

plt.show()

# --------------------------------------------------------------------------
# 13. Load the Saved Model
# --------------------------------------------------------------------------
# save_directory = "..." # replace with your directory
model_filename = "..."   # select the model with highest accuracy
model_path = os.path.join(save_directory, model_filename)

# If you compiled with custom metrics, pass them in custom_objects:
model = load_model(
    model_path,
    
)

model.summary()

# --------------------------------------------------------------------------
# 14. Grad-CAM Utilities
# --------------------------------------------------------------------------
def GradCam(model, img_array, layer_name, eps=1e-8):
    """
    Creates a Grad-CAM heatmap given a model, an image array, and a layer name.
    Args:
      model: A tf/keras model
      img_array: 4D batch array, e.g. shape (1, 224, 224, 3)
      layer_name: Name of the convolutional layer to inspect
      eps: Small value to avoid division by zero in normalization
    Returns:
      A 2D heatmap (float) of shape (height, width)
    """
    # Build a sub-model that outputs (layer_output, predictions)
    gradModel = Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )
    
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        # Here we pick the gradient wrt the first class index
        # If you have multiple classes, you can adapt accordingly.
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, convOutputs)
    
    # "Guided" part: keep only positive gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads

    # Remove batch dimension
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    # Compute weights for each channel
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    # Resize CAM to image size
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))

    # Normalize heatmap to [0,1]
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    
    return heatmap

def sigmoid(x, a, b, c):
    """
    Sigmoid-like function for optional emphasis in superimpose.
    """
    return c / (1 + np.exp(-a * (x-b)))

def superimpose(img_bgr, cam, thresh, emphasize=False):
    """
    Superimposes a Grad-CAM heatmap onto an image for visualization.
    Args:
      img_bgr: Original image (BGR) or (RGB), shape (H, W, 3)
      cam: 2D heatmap from Grad-CAM, shape (H, W)
      thresh: float threshold for emphasis
      emphasize: whether to apply a sigmoid transform to heatmap
    Returns:
      The heatmap-overlaid image in RGB format
    """
    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend heatmap and original image
    hif = 0.8
    superimposed_img = heatmap * hif + img_bgr.astype(np.float32)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb

# --------------------------------------------------------------------------
# 15. Identify the Last 2 Conv2D Layers in the Model
# --------------------------------------------------------------------------
conv2D_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]
print("All Conv2D layers:", conv2D_layers)

# This number changes based on the model architecture. We'll just take the 2 Conv layers
last_2_conv_layers = conv2D_layers[-2:]
print("Last 2 Conv2D layers:", last_2_conv_layers)

# --------------------------------------------------------------------------
# 16. Grad-CAM on a Sample Image
# --------------------------------------------------------------------------
image_path = r"..." # Replace with your own image path
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not read the image at {image_path}")

# Convert to RGB if needed
# (Some models expect RGB; if your model was trained on BGR images from OpenCV directly, skip.)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to match model's input shape
img_resized = cv2.resize(img_rgb, (224, 224))

# Preprocess the image for the model (e.g., normalize to [0,1])
img_preprocessed = img_resized / 255.0

# Expand dimensions to create a batch of 1
img_batch = np.expand_dims(img_preprocessed, axis=0)

# --------------------------------------------------------------------------
# 17. Plot Original and the 2 Grad-CAMs
# --------------------------------------------------------------------------
plt.figure(figsize=(20, 6))

# Plot the original image
plt.subplot(1, 6, 1)
plt.imshow(img_resized)
plt.axis('off')
plt.title('Original Image')

# Grad-CAM for each of the last 2 conv layers
for i, layer_name in enumerate(last_2_conv_layers, start=2):
    grad_cam_map = GradCam(model, img_batch, layer_name)
    superimposed_img = superimpose(img_resized, grad_cam_map, thresh=0.4, emphasize=True)

    plt.subplot(1, 6, i)
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title(f"Grad-CAM: {layer_name}")

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 18. Fusing Grad-CAM from All Convs
# -------------------------------------------------------------------------
def fuse_all_layers(conv2D_layers, model, img_bgr, emphasize=False):
    """
    Applies Grad-CAM to each convolutional layer and fuses them by averaging.

    Args:
        conv2D_layers: list of all convolutional layer names
        model: Keras model
        img_bgr: The original image (BGR) as NumPy array (H, W, 3)
        emphasize: bool, if True applies sigmoid transform in superimpose

    Returns:
        superimposed image (RGB) with fused heatmap
    """
    # Preprocess for model if needed
    img_resized = cv2.resize(img_bgr, (224, 224))       # match model input
    img_float = img_resized.astype(np.float32) / 255.0  # [0,1] normalization
    batch_input = np.expand_dims(img_float, axis=0)     # shape (1,224,224,3)

    # Collect Grad-CAM heatmaps for each conv layer
    cams = []
    for layer in conv2D_layers:
        cam = GradCam(model, batch_input, layer)
        # Resize CAM back to the original (pre-resized) image shape
        full_cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
        cams.append(full_cam)

    # Fuse by averaging across all conv layers
    fused_cam = np.mean(cams, axis=0)

    # Superimpose fused CAM onto original image
    superimposed = superimpose(img_bgr, fused_cam, thresh=0.5, emphasize=emphasize)
    return superimposed

# -------------------------------------------------------------------------
# 19. Example Usage: Multiple Images
# -------------------------------------------------------------------------
image_paths = [
    ...
]

# Show these images in a 12x2 grid: original + fused Grad-CAM
fig, axes = plt.subplots(len(image_paths), 2, figsize=(12, 4*len(image_paths)))
fig.subplots_adjust(wspace=0.02, hspace=0.02)

for idx, image_path in enumerate(image_paths):
    # Read image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # 1) Original
    axes[idx, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title("Original")

    # 2) Fused Grad-CAM
    fused_img = fuse_all_layers(conv2D_layers, model, img_bgr, emphasize=True)
    axes[idx, 1].imshow(fused_img)
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title("Fused Grad-CAM")

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# 20. Saliency Map Utilities
# --------------------------------------------------------------------------
def compute_saliency_map(model, img_batch):
    """
    Compute the saliency map for the top predicted class w.r.t. the input image.
    
    Args:
      model: Keras model
      img_batch: A 4D NumPy array, e.g. (1, 224, 224, 3)
    
    Returns:
      saliency: A 2D NumPy array (batch_size, height, width) representing
                the maximum absolute gradient for each pixel across channels.
    """
    img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        predictions = model(img_batch)  # shape: (batch_size, num_classes)

        # Find the index of the top predicted class for each image
        top_class = tf.argmax(predictions, axis=1)
        
        # Gather the score of the top class only
        # 'tf.one_hot' -> shape (batch_size, num_classes)
        top_class_score = tf.reduce_sum(
            predictions * tf.one_hot(top_class, predictions.shape[-1]),
            axis=-1
        )

    # Get the gradients of the top class score wrt the input image
    grads = tape.gradient(top_class_score, img_batch)

    # Take the maximum absolute gradient over the channels
    # grads.shape: (batch_size, H, W, C)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)  # (batch_size, H, W)

    return saliency.numpy()

def normalize_saliency_map(saliency_map):
    """
    Normalize the saliency map to [0, 1].
    """
    s_min = saliency_map.min()
    s_max = saliency_map.max()
    # Avoid division by zero
    if s_max - s_min < 1e-8:
        return saliency_map
    saliency_map -= s_min
    saliency_map /= (s_max - s_min)
    return saliency_map

# --------------------------------------------------------------------------
# 21. Visualization
# --------------------------------------------------------------------------
# We'll display 12 images, each in a row of 3 columns:
#  - Column 1: Original Image
#  - Column 2: Saliency Map (normalized)
#  - Column 3: Smoothed Saliency Map
fig, axes = plt.subplots(len(image_paths), 3, figsize=(18, 4*len(image_paths)))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for idx, image_path in enumerate(image_paths):
    # Load image via OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Resize to model's input (224x224)
    img_resized = cv2.resize(img, (224, 224))
    # Normalize to [0,1]
    img_preprocessed = img_resized.astype("float32") / 255.0

    # Compute saliency map (shape: (1, H, W))
    saliency_map = compute_saliency_map(model, np.expand_dims(img_preprocessed, axis=0))
    
    # Normalize
    normalized_saliency_map = normalize_saliency_map(saliency_map[0])  # shape (224, 224)

    # Apply Gaussian smoothing to highlight larger areas
    smoothed_saliency_map = gaussian_filter(normalized_saliency_map, sigma=2)

    # --- Plot Original Image ---
    axes[idx, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title('Original Image')

    # --- Plot Normalized Saliency ---
    axes[idx, 1].imshow(normalized_saliency_map, cmap='plasma')
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title('Saliency Map')

    # --- Plot Smoothed Saliency ---
    axes[idx, 2].imshow(smoothed_saliency_map, cmap='plasma')
    axes[idx, 2].axis('off')
    axes[idx, 2].set_title('Smoothed Saliency')

plt.tight_layout()
plt.show()
