import numpy as np

# x = np.zeros(shape=(4*7,3,2),dtype=int)
# print(x.shape)
# for i in range(0,4):
#     for j in range(0,7):
#         if j < 2:
#             x[i*7+j][0][0] = 5
# print(x)

x = [1,2,3,5,6]



def myfunc(myvar):
    myvar[2] = 10

myfunc(x)
print(x)
