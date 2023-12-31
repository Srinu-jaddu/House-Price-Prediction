import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy

def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return cost

def compute_gradient(X, y, w, b): 
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                   
    return dj_db, dj_dw

def gradient_descent(x, y, w_in, b_in, alpha, no_ite, compute_cost, compute_gradient):
    j_hist = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(no_ite):
        dj_db, dj_dw = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 10000:
            cost = compute_cost(x, y, w, b)
            j_hist.append(cost)
        if i % math.ceil(no_ite / 10) == 0:
            print(f"iteration {i} -- cost: {j_hist[i]}")
    return w, b, j_hist


#R2 Error
def r2_score(y_train, y_pred):
    y_mean = np.mean(y_train)
    ss_tot = np.sum((y_train - y_mean) ** 2)
    ss_res = np.sum((y_train - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

        

# Example data
x_train = np.array([[10.0, 2.0, 3.0, 4.0],
                   [20.0, 4.0, 6.0, 8.0],
                   [30.0, 6.0, 9.0, 12.0]])
y_train = np.array([100, 200, 300])

w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618], dtype=np.float64)
b_init = 785.1811367994083
initial_w = np.zeros_like(w_init, dtype=np.float64)
initial_b = 0.0

# Gradient descent settings
iterations = 10000
alpha = 0.001

# Run gradient descent
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b, alpha, iterations, compute_cost, compute_gradient)
print(f"b, w found by gradient descent: {b_final:.2f}, {w_final}\n")


#Prediction 
pred_y = []
for i in range(len(x_train)):
    s = np.dot(x_train[i],w_final) + b_final
    pred_y.append(s)
    print(f"Prediction {s} Actual Value {y_train[i]}\n")

#Accuracy 
print(f"R2 Score accuracy : {r2square(y_train,pred_y)}")
        
        
    

