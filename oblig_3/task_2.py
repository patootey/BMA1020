import numpy as np

#
# Answer to question a)
#
matA = np.zeros((10, 10))
matA[:, 4] = 1
matA[3:5, :] = 1
# add your two lines of code here
# the answer should be stored in the variable matA


#
# Answer to question b)
#
matB = np.zeros((10, 10))
matB[1:9, 1:9] = 1
# add your answer here. The answer should be stored in matB

#
# Answer to question c)
#
matC = matA + matB


#
# Answer to question d)
# ????
matD = matC > 1
