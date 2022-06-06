# https://github.com/Trotts/Siamese-Neural-Network-MNIST-Triplet-Loss/blob/main/Siamese-Neural-Network-MNIST.ipynb
# https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463

import os
import warnings
import numpy as np
import random
import matplotlib.pylab as plt
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from constants import path_outputs
from utilits.model_dense import create_embedding_model, create_SNN
from utilits.functions_for_seamnese_nn import (plot_triplets, create_batch, create_hard_batch, evaluate,
    data_generator, generate_prototypes, n_way_accuracy_prototypes, visualise_n_way_prototypes)

warnings.filterwarnings('ignore')

# сделаем так, чтобы tf не резервировал под себя сразу всю память
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# фиксируем сиды
# https://coderoad.ru/51249811/Воспроизводимые-результаты-в-Tensorflow-с-tf-set_random_seed
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)  # https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
tf.random.set_seed(seed)

# Use a custom non-dt name:
logdir = 'model_train_stats'
if not os.path.exists(logdir):
    os.mkdir(logdir)

"""
Import the data and reshape for use with the SNN
The data loaded in must be in the same format as tf.keras.datasets.mnist.load_data(), 
that is (x_train, y_train), (x_test, y_test)
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:5000]
y_test = y_test[:5000]
num_classes = len(np.unique(y_train))

x_train_w = x_train.shape[1]  # (60000, 28, 28)
x_train_h = x_train.shape[2]
x_test_w = x_test.shape[1]
x_test_h = x_test.shape[2]

x_train_w_h = x_train_w * x_train_h  # 28 * 28 = 784
x_test_w_h = x_test_w * x_test_h

x_train = np.reshape(x_train, (x_train.shape[0], x_train_w_h))/255.  # (60000, 784)
x_test = np.reshape(x_test, (x_test.shape[0], x_test_w_h))/255.

print(f'x:\ttrain: {x_train.shape},\ttest: {x_test.shape}\n'
      f'y:\ttrain: {y_train.shape},\ttest: {y_test.shape}\n'
      f'num_classes: {num_classes}\n'
      f'np.unique(y_test):\t{np.unique(y_test)}')

# Plotting the triplets
# plot_triplets([x_train[0], x_train[1], x_train[2]], x_train_w, x_train_h, 'Example of the triplet:')


"""
### Model Training Setup
FaceNet, the original triplet batch paper, 
draws a large random sample of triplets respecting the class distribution 
then picks N/2 hard and N/2 random samples (N = batch size), along with an `alpha` of 0.2
Logs out to Tensorboard, callback adapted from https://stackoverflow.com/a/52581175.
Saves best model only based on a validation loss. Adapted from https://stackoverflow.com/a/58103272.
"""

# Hyperparams
from constants import *
steps_per_epoch = int(x_train.shape[0]/batch_size)
val_steps = int(x_test.shape[0]/batch_size)

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size],y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)

with tf.device("/cpu:0"):
    # Create the embedding model
    network = create_embedding_model(emb_size, x_train_w_h)
    # Create the SNN
    siamese_net = create_SNN(network, x_train_w_h)
    # Compile the SNN
    optimiser_obj = Adam(lr=lr)
    siamese_net.compile(loss=triplet_loss, optimizer=optimiser_obj)

    # Store visualisations of the embeddings using PCA for display next to "after training" for comparisons
    num_vis = 500  # Take only the first num_vis elements of the test set to visualise
    embeddings_before_train = network.predict(x_test[:num_vis, :])
    pca = PCA(n_components=2)
    decomposed_embeddings_before = pca.fit_transform(embeddings_before_train)

# Display evaluation the untrained model
print("\nEvaluating the model without training for a baseline...\n")
evaluate(network, x_test, x_train, y_test, x_test_w_h, x_train_w, x_train_h, num_classes,
         DrawTestImage=False)


# Training logger
csv_log = os.path.join(logdir, 'training_history.csv')
csv_logger = CSVLogger(csv_log, separator=',', append=True)

"""
### Show example batches
Based on code found [here](https://zhangruochi.com/Create-a-Siamese-Network-with-Triplet-Loss-in-Keras/2020/08/11/).
# Display sample batches. This has to be performed after the embedding model is created
# as create_batch_hard utilises the model to see which batches are actually hard.
"""
# examples = create_batch(x_train, y_train, x_test, y_test, x_train_w_h, 1)
# plot_triplets(examples, x_train_w, x_train_h, "Example triplet batch:")
# ex_hard = create_hard_batch(network, x_train, y_train, x_test, y_test, x_train_w_h, 1, 1, split="train")
# plot_triplets(ex_hard, x_train_w, x_train_h, "Example semi-hard triplet batch:")

"""
### Training
Using `.fit(workers = 0)` fixes the error when using hard batches where TF can't predict on the embedding network
whilst fitting the siamese network (see: https://github.com/keras-team/keras/issues/5511#issuecomment-427666222).
Training:
"""
print(f"Starting training process!\n{'-' * 90}")

# Only save the best model weights based on the val_loss
checkpoint = ModelCheckpoint(os.path.join(logdir, 'snn_model.h5'),#'snn_model-{epoch:02d}-{val_loss:.2f}.h5'),
                             monitor='val_loss', verbose=1,
                             save_best_only=True,
                             # save_weights_only=True,
                             mode='auto')

# Save the embedding model weights if you save a new snn best model based on the model checkpoint above
callbacks = [csv_logger, checkpoint]

# Make the model work over the two GPUs we have
# num_gpus = get_num_gpus()
# parallel_snn = multi_gpu_model(siamese_net, gpus = num_gpus)
num_gpus = 1
snn = siamese_net
batch_per_gpu = int(batch_size / num_gpus)

snn.compile(loss=triplet_loss, optimizer=optimiser_obj)

history = snn.fit(
    data_generator(network, x_train, y_train, x_test, y_test, x_train_w_h,
                   emb_size, batch_size=batch_per_gpu, num_hard=num_hard),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, verbose=1, workers=1,
    callbacks=callbacks,
    validation_data=data_generator(network, x_train, y_train, x_test, y_test, x_train_w_h,
                                   emb_size, batch_size=batch_per_gpu, num_hard=num_hard, split="test"),
    validation_steps=val_steps)

plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title(f'bs={batch_size}, emb_size={emb_size}, lr={lr}, alpha={alpha}')
plt.grid()
plt.savefig(f'{path_outputs}/history_bs_{batch_size}_emb_size_{emb_size}_lr_{lr}_alpha_{alpha}.png')
plt.show()

print(f"{'-' * 90}\nTraining complete.")


"""
## Load in best trained SNN and emb model
# The best performing model weights has the higher epoch number due to only saving the best weights
"""

loaded_emb_model = tf.keras.models.load_model(os.path.join(logdir, 'snn_model.h5'),
                                              custom_objects={'triplet_loss': triplet_loss})
embeddings_after_train = network.predict(x_test[:num_vis, :])
# embeddings_after_train = loaded_emb_model.predict(x_test[:num_vis, :])
pca = PCA(n_components=2)
decomposed_embeddings_after = pca.fit_transform(embeddings_after_train)
evaluate(network, x_test, x_train, y_test, x_test_w_h, x_train_w, x_train_h, num_classes,
         DrawTestImage=False)

"""
### Comparisons of the embeddings in the latent space
Based on [this notebook](https://github.com/AdrianUng/keras-triplet-loss-mnist/blob/master/Triplet_loss_KERAS_semi_hard_from_TF.ipynb).
"""
step = 1  # Step = 1, take every element

dict_embeddings = {}
dict_gray = {}
test_class_labels = np.unique(np.array(y_test))

decomposed_embeddings_after = pca.fit_transform(embeddings_after_train)

fig = plt.figure(figsize=(16, 8))
for label in test_class_labels:
    y_test_labels = y_test[:num_vis]

    decomposed_embeddings_class_before = decomposed_embeddings_before[y_test_labels == label]
    decomposed_embeddings_class_after = decomposed_embeddings_after[y_test_labels == label]

    plt.subplot(1, 2, 1)
    plt.scatter(decomposed_embeddings_class_before[::step, 1], decomposed_embeddings_class_before[::step, 0],
                label=str(label))
    plt.title('Embedding Locations Before Training')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(decomposed_embeddings_class_after[::step, 1], decomposed_embeddings_class_after[::step, 0],
                label=str(label))
    plt.title('Embedding Locations After %d Training Epochs' % epochs)
    plt.legend()

plt.savefig(f'{path_outputs}/Embedding_Locations_After_{epochs}_Training Epochs'
            f'_bs_{batch_size}_emb_size_{emb_size}_lr_{lr}_alpha_{alpha}.png')
plt.show()


"""
### Determine n_way_accuracy
"""
prototypes = generate_prototypes(x_test, y_test, network)
n_way_accuracy_prototypes(val_steps, num_classes, network, x_test, y_test, prototypes)
"""
### Visualise support set inference
Based on code found [here](https://github.com/asagar60/One-Shot-Learning/blob/master/Omniglot_data/One_shot_implementation.ipynb).
"""
n_samples = 10
# Reduce the label set down from size n_classes to n_samples
labels = np.random.choice(np.unique(y_test), size=n_samples, replace=False)
for label in labels:

    # Find all images of the chosen test class
    imgs_of_label = np.where(y_test == label)[0]

    sample_imgs, min_index = visualise_n_way_prototypes(
        network, x_test, y_test, x_test_w_h, prototypes, label, labels)

    img_matrix = []
    for index in range(1, len(sample_imgs)):
        img_matrix.append(np.reshape(sample_imgs[index], (x_train_w, x_train_h)))

    img_matrix = np.asarray(img_matrix)
    img_matrix = np.vstack(img_matrix)

    f, ax = plt.subplots(1, 3, figsize=(10, 12))
    f.tight_layout()
    ax[0].imshow(np.reshape(sample_imgs[0], (x_train_w, x_train_h)),vmin=0, vmax=1,cmap='Greys')
    ax[0].set_title("Test Image")
    ax[1].imshow(img_matrix, vmin=0, vmax=1,cmap='Greys')
    ax[1].set_title("Support Set (Img of same class shown first)")
    ax[2].imshow(np.reshape(sample_imgs[min_index], (x_train_w, x_train_h)), vmin=0, vmax=1, cmap='Greys')
    ax[2].set_title("Image most similar\nto Test Image in Support Set")
    plt.savefig(f'{path_outputs}/Test_Images_for_label_{label}_after_{epochs}_Training Epochs'
                f'_bs_{batch_size}_emb_size_{emb_size}_lr={lr}_alpha_{alpha}.png')
    plt.show()
