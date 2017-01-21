# Experiment pipeline

### Results (Standart, lowFP, lowFN)

|            W    |           KNN            |       LOF        |        LoOP       |
|:---------------:|:------------------------:|:----------------:|:-----------------:|
|        W*       |  68.64,   61.07,   73.64 |                  |                   |
|      W*-CAD     |-307.86, -692.82, -176.74 |                  |                   |
|     W*-ICAD     |  60.39,   52.72,   67.30 |                  |                   |
|     W-ICAD*     |  60.39,   52.72,   67.30 |                  |                   |
| W*-offline ICAD |  44.85,    8.08,   58.92 |                  |                   |
|     W*-LCD      |                          |                  |                   |

### Average time (per value), ms

|            W    |          KNN        |       LOF        |        LoOP       |
|:---------------:|--------------------:|-----------------:|:-----------------:|
|        W*       |       59.285        |                  |                   |
|      W*-CAD     |    22337.588        |                  |                   |
|     W*-ICAD     |        1.368        |                  |                   |
| W*-offline ICAD |                     |                  |                   |
|     W*-LCD      |                     |                  |                   |

### Average time (per dataset), s

|            W    |          KNN        |       LOF        |        LoOP       |
|:---------------:|--------------------:|:----------------:|:-----------------:|
|        W*       |       373.654       |                  |                   |
|      W*-CAD     |    140787.657       |                  |                   |
|     W*-ICAD     |         8.624       |                  |                   |
| W*-offline ICAD |                     |                  |                   |
|     W*-LCD      |                     |                  |                   |


# Optional

* Matrix regularization
* Mahalanobis <-> weighted distance

