import numpy as np

kernels_all = [[] for i in range(5)]
num_cycle1 = [1, 2, 3, 4, 5]  

kernels_all2 = [[] for i in range(7)]
num_cycle2 = [1, 2, 3, 4, 5, 6, 7] 

kernels_all3 = [[] for i in range(1)]

kernels_all4 = [[] for i in range(1)]

def GenerateKernels():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle1: 
        kernels = []
        for j in range(i):  
            k_size = (2 * i) + 1  
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32)  
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2 
            red_size = (j + 1) * 2 - 1
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size 
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val 
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size) 
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val 

            kernels.append(kernel)
        kernels_all[i - 1] = kernels
        pass
    return kernels_all

def GenerateKernels2():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle2:  
        kernels = []
        for j in range(1): 
            k_size = (2 * i) + 1 
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32) 
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2  
            red_size = (j + 1) * 2 - 1 
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size  
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val  
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size) 
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val  
            kernels.append(kernel)
        kernels_all2[i - 1] = kernels
        pass
    return kernels_all2

def GenerateKernels3():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 9.0
    kernels_all3[0].append(kernel)
    return kernels_all3

def GenerateKernels4():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 8.0 * -1
    kernel[1, 1] = 0
    kernels_all4[0].append(kernel)
    return kernels_all4
