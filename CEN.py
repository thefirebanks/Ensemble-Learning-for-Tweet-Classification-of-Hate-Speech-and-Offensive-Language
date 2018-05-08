# This code calculates the Confusion Entropy measure CEN
# from https://www.sciencedirect.com/science/article/pii/S0957417409009828
#
# It assumes that the confusion matrix has the actual label for rows (first index)
# and predicted label for columns (second index)
#
# Adam Eck
# 04/10/18

import math

# this is P_{i,j}^i in the paper
# it is called first becaue the superscript is the first index
def calcPfirst(i, j, matrix):
    if i == j:
        return 0

    sum = 0
    for k in range(len(matrix)):
        sum += matrix[i][k] + matrix[k][i]

    return matrix[i][j] / sum

# this is P_{i,j}^j in the paper
# it is called second becaue the superscript is the second index
def calcPsecond(i, j, matrix):
    if i == j:
        return 0

    sum = 0
    for k in range(len(matrix)):
        sum += matrix[j][k] + matrix[k][j]

    return matrix[i][j] / sum

# this is P_j in the paper
def calcP(j, matrix):
    sum = 0
    denom = 0
    for k in range(len(matrix)):
        sum += matrix[j][k] + matrix[k][j]

        for l in range(len(matrix[k])):
            denom += matrix[k][l]

    return sum / (2 * denom)

# this is CEN_j in the paper
def calcCENj(j, matrix):
    N = len(matrix)
    base = 2 * (N - 1)
    sum = 0
    for k in range(N):
        if j == k:
            continue

        first = calcPfirst(j, k, matrix)
        second = calcPsecond(k, j, matrix)

        firstH = 0 if first == 0.0 else -first * math.log(first, base)
        secondH = 0 if second == 0.0 else -second * math.log(second, base)

        sum += firstH + secondH

    return sum

# this is the overall CEN measure
def calcCEN(matrix):
    sum = 0
    for j in range(len(matrix)):
        sum += calcP(j, matrix) * calcCENj(j, matrix)

    return sum

# the first four are the confusion matrices in Table 2ab and 3cd
# I used these to verify that I calculated the same results they had in Section 4.1
#matrix = [[70, 0, 0, 0], [5, 0, 0, 0], [10, 0, 0, 0], [15, 0, 0, 0]]
#matrix = [[70, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 10], [0, 15, 0, 0]]
#matrix = [[50, 5, 15, 0], [5, 0, 0, 0], [0, 0, 10, 0], [0, 5, 0, 10]]
#matrix = [[50, 20, 0, 0], [0, 5, 0, 0], [0, 10, 0, 0], [0, 0, 0, 15]]

# this last confusion matrix should have very high entropy
matrix = [[18, 18, 17, 17], [2, 1, 1, 1], [3, 3, 2, 2], [4, 4, 4, 3]]

# test the functions
#print(calcCEN(matrix))
