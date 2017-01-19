# Experiment pipeline

### Results (Standart, lowFP, lowFN)

|            W    |           KNN            |       LOF        |        LoOP       |
|:---------------:|:------------------------:|:----------------:|:-----------------:|
|        W*       |  68.64,   61.07,   73.64 |                  |                   |
|      W*-CAD     |-307.86, -692.82, -176.74 |                  |                   |
|     W*-ICAD     |                          |                  |                   |
| W*-offline ICAD |                          |                  |                   |
|     W*-LCD      |                          |                  |                   |

### Average time (per value), ms

|            W    |          KNN        |       LOF        |        LoOP       |
|:---------------:|--------------------:|-----------------:|:-----------------:|
|        W*       |       59.285        |                  |                   |
|      W*-CAD     |    22337.588        |                  |                   |
|     W*-ICAD     |                     |                  |                   |
| W*-offline ICAD |                     |                  |                   |
|     W*-LCD      |                     |                  |                   |

### Average time (per dataset), s

|            W    |          KNN        |       LOF        |        LoOP       |
|:---------------:|--------------------:|:----------------:|:-----------------:|
|        W*       |       373.654       |                  |                   |
|      W*-CAD     |    140787.657       |                  |                   |
|     W*-ICAD     |                     |                  |                   |
| W*-offline ICAD |                     |                  |                   |
|     W*-LCD      |                     |                  |                   |


# Optional

* Matrix regularization
* Mahalanobis <-> weighted distance

