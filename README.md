# Experiment pipeline

### Results (Standart, lowFP, lowFN)

|            W    |           KNN     |        LOF        |
|:---------------:|:-----------------:|:-----------------:|
|        W*       | 65.48 58.04 71.16 | 15.60 08.65 15.87 |
|     W*-ICAD     | 59.84 52.18 65.62 | 25.62 19.80 29.97 |
| W*-offline ICAD | 50.90 22.15 62.38 | 28.57 07.56 37.15 |
|     W*-LCD      | 41.47 07.83 54.37 | 18.19 00.00 30.23 |

### Average time (per value), ms

|            W    |   KNN  |   LOF  |
|:---------------:|-------:|-------:|
|        W*       | 14.955 | 16.820 |
|     W*-ICAD     |  0.628 |  0.733 |
| W*-offline ICAD |  0.205 |  0.335 |
|     W*-LCD      |  0.990 |  1.100 |

### Average time (per dataset), s

|            W    |  KNN  |  LOF   |
|:---------------:|------:|-------:|
|        W*       | 94.26 | 106.01 |
|     W*-ICAD     |  3.96 |   4.62 |
| W*-offline ICAD |  1.29 |   2.11 |
|     W*-LCD      |  6.24 |   6.93 |


# Optional

* Matrix regularization
* Mahalanobis <-> weighted distance

