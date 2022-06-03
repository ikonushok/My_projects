# Verification of several ways to solve the problem of classification of aircraft maneuvers
___
## Methods:
1. Clusterization by TimeSeriesKMeans_cluster_analisys: `clusterization.py`
2. Clusterization: `dynamic_time_warping.py`
3. Neural Networks: 
   1. [Siamese Neural Network](https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463)
   2. [Music Track Classification](https://github.com/ikonushok/My_projects/tree/main/Sound)
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