#%%
import numpy as np


A = np.matrix(  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ,[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

count = np.sum(A, axis=0)
n = np.argmax(count)
print(A[:, 3])
# I = np.matrix(np.eye(A.shape[0]))
# A_hat = A + I

# D = np.array(np.sum(A_hat, axis=0))[0]
# D = np.matrix(np.diag(D))
# X = np.matrix([
#                 [1]
#                 for i in range(A.shape[0])], dtype=float)
# print(D**-1 * A_hat * X)
# print(D)

# D = np.matrix(np.diag(D))
# %%
