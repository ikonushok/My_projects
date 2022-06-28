
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.client import device_lib

from sklearn.metrics import roc_curve, roc_auc_score


# фиксируем сиды
# https://coderoad.ru/51249811/Воспроизводимые-результаты-в-Tensorflow-с-tf-set_random_seed
path_outputs = 'outputs'
source_root = 'source_root'

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)  # https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
tf.random.set_seed(seed)


def plot_triplets(examples, x_train_w, x_train_h, title=' '):
    plt.figure(figsize=(6, 2))
    for i in range(3):
        if i == 2:
            plt.title(title)
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.reshape(examples[i], (x_train_w, x_train_h)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'{path_outputs}/triplets.png')
    plt.show()


"""
Create triplet batches
Random batches are generated by create_batch. Semi-hard triplet batches are generated by create_batch_hard.

Semi-Hard: dist(A, P) < dist(A, N) < dist(A, P) + margin. Using only easy triplets will lead to no learning. 
Hard triplets generate high loss and have high impact on training parameters, 
but may cause any mislabelled data to cause too much of a weight change.
"""
def create_batch(x_train, y_train, x_test, y_test, x_train_w_h, batch_size=256, split="train"):
    x_anchors = np.zeros((batch_size, x_train_w_h))
    x_positives = np.zeros((batch_size, x_train_w_h))
    x_negatives = np.zeros((batch_size, x_train_w_h))

    if split == "train":
        data = x_train
        data_y = y_train
    else:
        data = x_test
        data_y = y_test

    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, data.shape[0] - 1)
        x_anchor = data[random_index]
        y = data_y[random_index]

        indices_for_pos = np.squeeze(np.where(data_y == y))
        indices_for_neg = np.squeeze(np.where(data_y != y))

        x_positive = data[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = data[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]

        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative

    return [x_anchors, x_positives, x_negatives]



def create_hard_batch(network, x_train, y_train, x_test, y_test, x_train_w_h, batch_size, num_hard, split="train"):
    x_anchors = np.zeros((batch_size, x_train_w_h))
    x_positives = np.zeros((batch_size, x_train_w_h))
    x_negatives = np.zeros((batch_size, x_train_w_h))

    # if split == "train":
    #     data = x_train
    #     data_y = y_train
    # else:
    #     data = x_test
    #     data_y = y_test

    # Generate num_hard number of hard examples:
    hard_batches = []
    batch_losses = []
    rand_batches = []
    # Get some random batches
    for i in range(0, batch_size):
        hard_batches.append(create_batch(x_train, y_train, x_test, y_test, x_train_w_h, 1, split))

        A_emb = network.predict(hard_batches[i][0])
        P_emb = network.predict(hard_batches[i][1])
        N_emb = network.predict(hard_batches[i][2])

        # Compute d(A, P) - d(A, N) for each selected batch
        batch_losses.append(np.sum(np.square(A_emb - P_emb), axis=1) - np.sum(np.square(A_emb - N_emb), axis=1))

    # Sort batch_loss by distance, highest first, and keep num_hard of them
    hard_batch_selections = [x for _, x in sorted(zip(batch_losses, hard_batches), key=lambda x: x[0])]
    hard_batches = hard_batch_selections[:num_hard]

    # Get batch_size - num_hard number of random examples
    num_rand = batch_size - num_hard
    for i in range(0, num_rand):
        rand_batch = create_batch(x_train, y_train, x_test, y_test, x_train_w_h, 1, split)
        rand_batches.append(rand_batch)

    selections = hard_batches + rand_batches

    for i in range(0, len(selections)):
        x_anchors[i] = selections[i][0]
        x_positives[i] = selections[i][1]
        x_negatives[i] = selections[i][2]

    return [x_anchors, x_positives, x_negatives]



"""
### Create the Triplet Loss Function
"""
def triplet_loss(y, emb_size, alpha=0.2):
    print(y.shape)
    print(emb_size)
    anchor, positive, negative = y[:,:emb_size], y[:,emb_size:2*emb_size], y[:,2*emb_size:]
    print(anchor, positive, negative)
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)



"""
### Data Generator
This function creates hard batches for the network to train on. `y` is required by TF but not by our model, 
so just return a filler to keep TF happy.
"""
def data_generator(network, x_train, y_train, x_test, y_test, x_train_w_h,
                   emb_size, batch_size=256, num_hard=50, split="train"):
    while True:
        x = create_hard_batch(network, x_train, y_train, x_test, y_test, x_train_w_h, batch_size, num_hard, split)
        y = np.zeros((batch_size, 3*emb_size))
        yield x, y


"""
### Evaluation
Allows
for the model's metrics to be visualised and evaluated. 
Based on [this Medium post](https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352) 
and [this GitHub notebook](https://github.com/asagar60/One-Shot-Learning/blob/master/Omniglot_data/One_shot_implementation.ipynb).
"""
def compute_dist(a, b):
    return np.linalg.norm(a - b)

def compute_probs(network, X, Y):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,w,h,1) containing pics to evaluate
        Y : tensor of shape (m,) containing true class

    Returns
        probs : array of shape (m,m) containing distances

    '''
    m = X.shape[0]
    nbevaluation = int(m * (m - 1) / 2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))

    # Compute all embeddings for all imgs with current embedding network
    embeddings = network.predict(X)

    k = 0

    # For each img in the evaluation set
    for i in range(m):
        # Against all other images
        for j in range(i + 1, m):
            # compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
            probs[k] = -compute_dist(embeddings[i, :], embeddings[j, :])
            if (Y[i] == Y[j]):
                y[k] = 1
                # print("{3}:{0} vs {1} : \t\t\t{2}\tSAME".format(i,j,probs[k],k, Y[i], Y[j]))
            else:
                y[k] = 0
                # print("{3}:{0} vs {1} : {2}\tDIFF".format(i,j,probs[k],k, Y[i], Y[j]))
            k += 1
    return probs, y


def compute_metrics(probs, yprobs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)

    return fpr, tpr, thresholds, auc


def draw_roc(fpr, tpr, thresholds, auc):
    # find threshold
    targetfpr = 1e-3
    _, idx = find_nearest(fpr, targetfpr)
    threshold = thresholds[idx]
    recall = tpr[idx]

    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC: {0:.3f}   Sensitivity : {2:.1%}   @FPR={1:.0e}\n'
              'Threshold={3}'.format(auc, targetfpr, recall, abs(threshold)))
    plt.savefig(f'{path_outputs}/AUC_{round(auc, 2)}_Sensitivity_{targetfpr}'
                f'_FPR_{round(recall, 3)}_Threshold_{round(abs(threshold), 2)}.png')
    plt.show()


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1], idx - 1
    else:
        return array[idx], idx


def draw_interdist(network, num_classes, x_test_w_h, x_test, epochs):
    interdist = compute_interdist(network, num_classes, x_test_w_h, x_test)

    data = []
    for i in range(num_classes):
        data.append(np.delete(interdist[i, :], [i]))

    fig, ax = plt.subplots()
    ax.set_title('Evaluating embeddings distance from each other after {0} epochs'.format(epochs))
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    ax.boxplot(data, showfliers=False, showbox=True)
    locs, labels = plt.xticks()
    plt.xticks(locs, np.arange(num_classes))
    plt.savefig(f'{path_outputs}/Evaluating_embeddings_distance_from_each_other_after_{epochs}_epochs.png')
    plt.show()


def compute_interdist(network, num_classes, x_test_w_h, x_test):
    '''
    Computes sum of distances between all classes embeddings on our reference test image:
        d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
        A good model should have a large distance between all theses embeddings

    Returns:
        array of shape (num_classes,num_classes)
    '''
    res = np.zeros((num_classes, num_classes))
    ref_images = np.zeros((num_classes, x_test_w_h))

    # generates embeddings for reference images
    for i in range(num_classes):
        ref_images[i, :] = x_test[i]

    ref_embeddings = network.predict(ref_images)

    for i in range(num_classes):
        for j in range(num_classes):
            res[i, j] = math.dist(ref_embeddings[i], ref_embeddings[j])
    return res


def DrawTestImage(network, images, num_classes, x_test_w_h, x_test, y_test, x_train_w, x_train_h, refidx=0):
    '''
    Evaluate some pictures vs some samples in the test set image must be of shape(1,w,h,c)
    Returns scores : result of the similarity scores with the basic images => (N)
    '''
    nbimages = images.shape[0]

    # generates embedings for given images
    image_embedings = network.predict(images)

    # generates embedings for reference images
    ref_images = np.zeros((num_classes, x_test_w_h))
    for i in range(num_classes):
        images_at_this_index_are_of_class_i = np.squeeze(np.where(y_test == i))
        ref_images[i, :] = x_test[images_at_this_index_are_of_class_i[refidx]]

    ref_embedings = network.predict(ref_images)

    for i in range(nbimages):
        # Prepare the figure
        fig = plt.figure(figsize=(16, 2))
        subplot = fig.add_subplot(1, num_classes + 1, 1)
        plt.axis("off")
        plotidx = 2

        # Draw this image
        plt.imshow(np.reshape(images[i], (x_train_w, x_train_h)), vmin=0, vmax=1, cmap='Greys')
        subplot.title.set_text("Test image")

        for ref in range(num_classes):
            # Compute distance between this images and references
            dist = compute_dist(image_embedings[i, :], ref_embedings[ref, :])
            # Draw
            subplot = fig.add_subplot(1, num_classes + 1, plotidx)
            plt.axis("off")
            plt.imshow(np.reshape(ref_images[ref, :], (x_train_w, x_train_h)), vmin=0, vmax=1, cmap='Greys')
            subplot.title.set_text(("Class {0}\n{1:.3e}".format(ref, dist)))
            plotidx += 1


def generate_prototypes(x_data, y_data, network):
    classes = np.unique(y_data)
    prototypes = {}

    for c in classes:
        # c = classes[0]
        # Find all images of the chosen test class
        locations_of_c = np.where(y_data == c)[0]

        imgs_of_c = x_data[locations_of_c]
        # print(f'function.py 341:\t{imgs_of_c.shape}')
        imgs_of_c_embeddings = network.predict(imgs_of_c)

        # Get the median of the embeddings to generate a prototype for the class (reshaping for PCA)
        prototype_for_c = np.median(imgs_of_c_embeddings, axis=0).reshape(1, -1)
        # Add it to the prototype dict
        prototypes[c] = prototype_for_c

    return prototypes


def test_one_shot_prototypes(network, sample_embeddings):
    distances_from_img_to_test_against = []
    # As the img to test against is in index 0, we compare distances between img@0 and all others
    for i in range(1, len(sample_embeddings)):
        distances_from_img_to_test_against.append(compute_dist(sample_embeddings[0], sample_embeddings[i]))
    # As the correct img will be at distances_from_img_to_test_against index 0 (sample_imgs index 1),
    # If the smallest distance in distances_from_img_to_test_against is at index 0,
    # we know the one shot test got the right answer
    is_min = distances_from_img_to_test_against[0] == min(distances_from_img_to_test_against)
    is_max = distances_from_img_to_test_against[0] == max(distances_from_img_to_test_against)
    return int(is_min and not is_max)


def n_way_accuracy_prototypes(n_val, n_way, network, x_test, y_test, prototypes):
    num_correct = 0

    for val_step in range(n_val):
        num_correct += load_one_shot_test_batch_prototypes(n_way, network, x_test, y_test, prototypes)

    accuracy = num_correct / n_val * 100

    return accuracy


def load_one_shot_test_batch_prototypes(n_way, network, x_test, y_test, prototypes):
    labels = np.unique(y_test)
    # Reduce the label set down from size n_classes to n_samples
    labels = np.random.choice(labels, size=n_way, replace=False)

    # Choose a class as the test image
    label = random.choice(labels)
    # Find all images of the chosen test class
    imgs_of_label = np.where(y_test == label)[0]

    # Randomly select a test image of the selected class, return it's index
    img_of_label_idx = random.choice(imgs_of_label)

    # Expand the array at the selected indexes into useable images
    img_of_label = np.expand_dims(x_test[img_of_label_idx], axis=0)

    sample_embeddings = []
    # Get the anchor image embedding
    anchor_prototype = network.predict(img_of_label)
    sample_embeddings.append(anchor_prototype)

    # Get the prototype embedding for the positive class
    positive_prototype = prototypes[label]

    sample_embeddings.append(positive_prototype)

    # Get the negative prototype embeddings
    # Remove the selected test class from the list of labels based on it's index
    label_idx_in_labels = np.where(labels == label)[0]
    other_labels = np.delete(labels, label_idx_in_labels)

    # Get the embedding for each of the remaining negatives
    for other_label in other_labels:
        negative_prototype = prototypes[other_label]
        sample_embeddings.append(negative_prototype)

    correct = test_one_shot_prototypes(network, sample_embeddings)

    return correct


def visualise_n_way_prototypes(network, x_test, y_test, x_test_w_h, prototypes, label, labels):

    imgs_of_label = np.where(y_test == label)[0]

    # Randomly select a test image of the selected class, return it's index
    img_of_label_idx = random.choice(imgs_of_label)

    # Get another image idx that we know is of the test class for the sample set
    label_sample_img_idx = random.choice(imgs_of_label)

    # Expand the array at the selected indexes into useable images
    img_of_label = np.expand_dims(x_test[img_of_label_idx], axis=0)
    label_sample_img = np.expand_dims(x_test[label_sample_img_idx], axis=0)

    # Make the first img in the sample set the chosen test image, the second the other image
    sample_imgs = np.empty((0, x_test_w_h))
    sample_imgs = np.append(sample_imgs, img_of_label, axis=0)
    sample_imgs = np.append(sample_imgs, label_sample_img, axis=0)

    sample_embeddings = []

    # Get the anchor embedding image
    anchor_prototype = network.predict(img_of_label)
    sample_embeddings.append(anchor_prototype)

    # Get the prototype embedding for the positive class
    positive_prototype = prototypes[label]
    sample_embeddings.append(positive_prototype)

    # Get the negative prototype embeddings
    # Remove the selected test class from the list of labels based on it's index
    label_idx_in_labels = np.where(labels == label)[0]
    other_labels = np.delete(labels, label_idx_in_labels)
    # Get the embedding for each of the remaining negatives
    for other_label in other_labels:
        negative_prototype = prototypes[other_label]
        sample_embeddings.append(negative_prototype)

        # Find all images of the other class
        imgs_of_other_label = np.where(y_test == other_label)[0]
        # Randomly select an image of the selected class, return it's index
        another_sample_img_idx = random.choice(imgs_of_other_label)
        # Expand the array at the selected index into useable images
        another_sample_img = np.expand_dims(x_test[another_sample_img_idx], axis=0)
        # Add the image to the support set
        sample_imgs = np.append(sample_imgs, another_sample_img, axis=0)

    distances_from_img_to_test_against = []

    # As the img to test against is in index 0, we compare distances between img@0 and all others
    for i in range(1, len(sample_embeddings)):
        distances_from_img_to_test_against.append(compute_dist(sample_embeddings[0], sample_embeddings[i]))

    # + 1 as distances_from_img_to_test_against doesn't include the test image
    min_index = distances_from_img_to_test_against.index(min(distances_from_img_to_test_against)) + 1

    return sample_imgs, min_index


def evaluate(network, x_test, x_train, y_test, x_test_w_h, x_train_w, x_train_h, num_classes,
             DrawTestImage=False, epochs=0):
    probs, yprob = compute_probs(network, x_test[:500, :], y_test[:500])
    fpr, tpr, thresholds, auc = compute_metrics(probs, yprob)
    draw_roc(fpr, tpr, thresholds, auc)
    draw_interdist(network, num_classes, x_test_w_h, x_test, epochs)

    if DrawTestImage == True:
        for i in range(3):
            DrawTestImage(network, np.expand_dims(x_train[i], axis=0),
                          num_classes, x_test_w_h, x_test, y_test, x_train_w, x_train_h)

def get_num_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])