# train_asl.py
import os
import math
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------------------
# CONFIG
# -------------------
DATA_DIR = "data/asl_alphabet_train"   # path to train folders (each folder is a class)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
SEED = 1337
NUM_CLASSES = 29
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS_HEAD = 6
EPOCHS_FINETUNE = 10
PATIENCE = 3
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------
# UTILS
# -------------------
def get_label_names(data_dir):
    # alphabetical sorted class folder names
    classes = sorted([p.name for p in pathlib.Path(data_dir).iterdir() if p.is_dir()])
    return classes

CLASS_NAMES = get_label_names(DATA_DIR)
print("Classes:", CLASS_NAMES)

# -------------------
# BUILD tf.data
# -------------------
def decode_and_resize(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# data listing
file_paths = []
labels = []
for i, cls in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(DATA_DIR, cls)
    for fn in os.listdir(cls_dir):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_paths.append(os.path.join(cls_dir, fn))
            labels.append(i)
file_paths = np.array(file_paths)
labels = np.array(labels)

# shuffle and split (80% train, 10% val, 10% test)
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(file_paths))
file_paths = file_paths[perm]
labels = labels[perm]

n = len(file_paths)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
train_files, train_labels = file_paths[:n_train], labels[:n_train]
val_files, val_labels = file_paths[n_train:n_train+n_val], labels[n_train:n_train+n_val]
test_files, test_labels = file_paths[n_train+n_val:], labels[n_train+n_val:]

def make_dataset(files, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(files), seed=SEED)
    ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x,y: (data_augment(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# -------------------
# AUGMENTATION
# -------------------
data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
])

train_ds = make_dataset(train_files, train_labels, training=True)
val_ds = make_dataset(val_files, val_labels, training=False)
test_ds = make_dataset(test_files, test_labels, training=False)

# -------------------
# MODEL: transfer learning with EfficientNetV2B0
# -------------------
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False, input_shape=(*IMAGE_SIZE,3), weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE,3))
x = inputs
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)
x = tf.keras.layers.Dense(512, activation="swish")(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# compile for head training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------
# CALLBACKS
# -------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_head.h5"),
                                       save_best_only=True, monitor="val_accuracy", mode="max"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
]

# -------------------
# TRAIN HEAD
# -------------------
history_head = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

# -------------------
# FINE-TUNE: unfreeze top layers of base_model
# -------------------
base_model.trainable = True
# freeze lower layers and unfreeze last N layers (tune N)
N_UNFREEZE = 40
for i, layer in enumerate(base_model.layers[:-N_UNFREEZE]):
    layer.trainable = False
for layer in base_model.layers[-N_UNFREEZE:]:
    layer.trainable = True

# lower LR for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_fine = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_finetuned.h5"),
                                       save_best_only=True, monitor="val_accuracy", mode="max"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
]

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, callbacks=callbacks_fine)

# -------------------
# EVALUATION on test set
# -------------------
model.save(os.path.join(MODEL_DIR, "final_model_saved"))
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# predict and show classification report
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# build true labels in same order as test_ds batches
y_true = np.concatenate([y for x,y in test_ds], axis=0)

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# confusion matrix (optionally plot for top confusing classes)
cm = confusion_matrix(y_true, y_pred)
np.save(os.path.join(MODEL_DIR, "confusion_matrix.npy"), cm)

# minimal plot of loss/acc
def plot_hist(h, title):
    plt.figure(figsize=(8,4))
    plt.plot(h.history.get("loss", []), label="loss")
    plt.plot(h.history.get("val_loss", []), label="val_loss")
    plt.title(title); plt.legend()

plot_hist(history_head, "head training")
plot_hist(history_ft, "finetune training")
plt.show()

# Save mapping
import json
with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(CLASS_NAMES, f)
print("Saved model and artifacts to", MODEL_DIR)
