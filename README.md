# Experiment pipeline

### Results (Standart, lowFP, lowFN)

|            W    |           KNN            |       LOF        |
|:---------------:|:------------------------:|:----------------:|
|        W*       |  68.64,   61.07,   73.64 |                  |
|     W*-ICAD     |  60.39,   52.72,   67.30 |                  |
| W*-offline ICAD |  44.85,   08.08,   58.92 |                  |
|     W*-LCD      |   |                  |
|     W-slice*    |  59.07,   49.84,   67.83 |                  |

### Average time (per value), ms

|            W    |          KNN        |       LOF        |
|:---------------:|--------------------:|-----------------:|
|        W*       |       59.285        |                  |
|     W*-ICAD     |        1.368        |                  |
| W*-offline ICAD |        0.561        |                  |
|     W*-LCD      |   |                  |                   |
|     W-slice*    |       20.720        |                  |

### Average time (per dataset), s

|            W    |          KNN        |       LOF        |
|:---------------:|--------------------:|:----------------:|
|        W*       |       373.654       |                  |
|     W*-ICAD     |         8.624       |                  |
| W*-offline ICAD |         3.534       |                  |
|     W*-LCD      |   |                  |
|     W-slice*    |       130.591       |                  |


# Optional

* Matrix regularization
* Mahalanobis <-> weighted distance

