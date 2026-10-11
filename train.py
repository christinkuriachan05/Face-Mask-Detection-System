import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "mask_detector.keras")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# FIND IMAGES RECURSIVELY
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
)


with_mask_files = []
without_mask_files = []


print("\n==============================================")
print("FACE MASK DETECTION TRAINING")
print("==============================================")

print("\nSearching dataset:")
print(os.path.abspath(DATASET_DIR))


if not os.path.exists(DATASET_DIR):

    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )


# Search EVERY folder inside dataset
for root, dirs, files in os.walk(DATASET_DIR):

    for filename in files:

        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        full_path = os.path.join(root, filename)

        # Look at the folder name containing the image
        folder_name = os.path.basename(root).lower()

        if folder_name == "with_mask":

            with_mask_files.append(full_path)

        elif folder_name == "without_mask":

            without_mask_files.append(full_path)


# Remove duplicates
with_mask_files = list(set(with_mask_files))
without_mask_files = list(set(without_mask_files))


# ============================================================
# SHOW DATASET INFORMATION
# ============================================================

print("\n==============================================")
print("DATASET CHECK")
print("==============================================")

print("With mask images    :", len(with_mask_files))
print("Without mask images :", len(without_mask_files))


if len(with_mask_files) == 0:

    raise ValueError(
        "\nNO WITH-MASK IMAGES FOUND!\n"
        "Make sure your images are inside a folder named:\n"
        "with_mask"
    )


if len(without_mask_files) == 0:

    raise ValueError(
        "\nNO WITHOUT-MASK IMAGES FOUND!\n"
        "Make sure your images are inside a folder named:\n"
        "without_mask"
    )


print("\nExample WITH MASK:")
print(with_mask_files[0])

print("\nExample WITHOUT MASK:")
print(without_mask_files[0])


# ============================================================
# CREATE LABELS
# ============================================================

# IMPORTANT:
#
# 0 = WITH MASK
# 1 = WITHOUT MASK
#

all_files = (
    with_mask_files +
    without_mask_files
)

all_labels = (
    [0] * len(with_mask_files) +
    [1] * len(without_mask_files)
)


# Shuffle together
combined = list(zip(all_files, all_labels))

random.shuffle(combined)

all_files, all_labels = zip(*combined)

all_files = list(all_files)
all_labels = list(all_labels)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

total = len(all_files)

validation_count = int(total * 0.20)

train_files = all_files[:-validation_count]
train_labels = all_labels[:-validation_count]

validation_files = all_files[-validation_count:]
validation_labels = all_labels[-validation_count:]


print("\n==============================================")
print("TRAINING DATA")
print("==============================================")

print("Total images      :", total)
print("Training images   :", len(train_files))
print("Validation images :", len(validation_files))


print("\nLabel meaning:")
print("0 = WITH MASK")
print("1 = WITHOUT MASK")


# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    label = tf.cast(
        label,
        tf.float32
    )

    return image, label


# ============================================================
# CREATE TF DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_files, train_labels)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validation_files, validation_labels)
)


train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    )

])


# ============================================================
# MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    augmentation,

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        256,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        1,
        activation="sigmoid"
    )

])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# DATASET BATCHING
# ============================================================

train_dataset = (
    train_dataset
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    validation_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

with_count = len(with_mask_files)
without_count = len(without_mask_files)

total_count = with_count + without_count

class_weight = {

    0: total_count / (2 * with_count),

    1: total_count / (2 * without_count)

}


print("\n==============================================")
print("CLASS WEIGHTS")
print("==============================================")

print(class_weight)


# ============================================================
# TRAIN
# ============================================================

print("\n==============================================")
print("STARTING TRAINING")
print("==============================================\n")


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weight

)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


model.save(
    MODEL_PATH
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n==============================================")
print("TRAINING COMPLETE")
print("==============================================")

print("\nModel saved:")
print(os.path.abspath(MODEL_PATH))


print("\n==============================================")
print("FINAL MODEL TEST")
print("==============================================")


# Test a few WITH MASK images
print("\nWITH MASK TEST:")

for path in with_mask_files[:5]:

    image = load_img(
        path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    image = img_to_array(image)

    image = image.astype(
        "float32"
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = float(
        model.predict(
            image,
            verbose=0
        )[0][0]
    )

    print(
        os.path.basename(path),
        "->",
        round(prediction, 4)
    )


# Test a few WITHOUT MASK images
print("\nWITHOUT MASK TEST:")

for path in without_mask_files[:5]:

    image = load_img(
        path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    image = img_to_array(image)

    image = image.astype(
        "float32"
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = float(
        model.predict(
            image,
            verbose=0
        )[0][0]
    )

    print(
        os.path.basename(path),
        "->",
        round(prediction, 4)
    )


print("\n==============================================")
print("DONE")
print("==============================================")

print("\nExpected:")
print("WITH MASK    -> values closer to 0")
print("WITHOUT MASK -> values closer to 1")

print("\nNow run:")
print("python app.py")