# Verification of several ways to solve the problem of classification of aircraft maneuvers
___
## Methods:
1. Clusterization by TimeSeriesKMeans_cluster_analisys: [clusterization.py](https://github.com/ikonushok/My_projects/blob/main/Tabular%20Data/Aircraft_maneuvers_classification/clusterization.py)
2. Clusterization by Dynamic Time Warping algorithm: [dynamic_time_warping.py](https://github.com/ikonushok/My_projects/blob/main/Tabular%20Data/Aircraft_maneuvers_classification/dynamic_time_warping.py)
then we can solve the problem with Classic ML
3. Here we have two types of Neural Networks: 
   1. [seamnese_nn.py](https://github.com/ikonushok/My_projects/blob/main/Tabular%20Data/Aircraft_maneuvers_classification/seamnese_nn.py) original from [Siamese Neural Network](https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463)
   2. [simple_nn.py](https://github.com/ikonushok/My_projects/blob/main/Tabular%20Data/Aircraft_maneuvers_classification/simple_nn.py)
___
## Classes:
-1- not labeled/undefined

0. phone in hands
1. aircraft engine is off
2. aircraft engine is on (no motion)
3. aircraft is moving on the ground
4. aircraft is taking off
5. cruise
6. aircraft is landing
7. turn (вираж) 15 deg, left
8. turn (вираж) 15 deg, right
9. turn (вираж) 30 deg, left
10. turn (вираж) 30 deg, right
11. turn (вираж) 45 deg, left
12. turn (вираж) 45 deg, right
13. акробатика
14. проход


---

## Siamese Neural Network with Triplet Loss trained on MNIST by Cameron Trotter
c.trotter2@ncl.ac.uk

This notebook builds an SNN to determine similarity scores between MNIST digits using a triplet loss function. 
The use of class prototypes at inference time is also explored. 

This notebook is based heavily on the approach described in 
[this Coursera course](https://www.coursera.org/learn/siamese-network-triplet-loss-keras/), 
which in turn is based on the [FaceNet](https://arxiv.org/abs/1503.03832) paper. 
Any uses of open-source code are linked throughout where utilised. 

For an in-depth guide to understand this code, and the theory behind it, please see 
[my article for Towards Data Science](https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463).

---
[How To Train Your Siamese Neural Network](https://github.com/Trotts/Siamese-Neural-Network-MNIST-Triplet-Loss/blob/main/Siamese-Neural-Network-MNIST.ipynb)

[Siamese Neural Network with Triplet Loss trained on MNIST](https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463)