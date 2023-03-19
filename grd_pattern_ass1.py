from numpy import *
import matplotlib.pyplot as plt
import copy
# Load our data set
x_train =array([16.99, 10.34,21.01,23.68,24.59,25.29,8.77,26.88,15.04,14.78,10.27])   #features
y_train =array([2.0, 3.0,3.0,2.0,4.0,4.0,2.0,4.0,2.0,2.0,2.0])   #target value


# Function to calculate the cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """
    m=x.shape[0]
    wj=0 #e gradient of the cost w.r.t. the parameters w
    bj=0 #e gradient of the cost w.r.t. the parameters b
    for i in range(m):
        fwb=w*x[i]+b #w & b are the paramenter
        wj_i=(fwb-y[i])*x[i]
        bj_i=fwb-y[i]
        bj+=bj_i#b=b+b_i
        wj+=wj_i
    wj=wj/m
    bj=bj/m
    return wj,bj

    # # Number of training examples
    # m = x.shape[0]
    # dj_dw = 0
    # dj_db = 0
    #
    # for i in range(m):
    #     f_wb = w * x[i] + b
    #     dj_dw_i = (f_wb - y[i]) * x[i]
    #     dj_db_i = f_wb - y[i]
    #     dj_db += dj_db_i#dj_db =dj_db+dj_db_i
    #     dj_dw += dj_dw_i
    # dj_dw = dj_dw / m
    # dj_db = dj_db / m
    #
    # return dj_dw, dj_db

a=compute_gradient(x_train,y_train,0,0)
print(a)
# plt_gradients(x_train,y_train, compute_cost, compute_gradient)
# plt.show()

# gradient descent implementation to make more then one iteration

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function: function
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] """
    w = copy.deepcopy(w_in)  # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = [] #cost J
    p_history = [] # w's at each iteration
    b = b_in #b_in initial values of model parameters
    w = w_in #w_in initial values of model parameters

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db # returned from the gradient_function is on the last dj_db/m
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # return w and J,w history for graphing


# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, J_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
#gradient_descent return w,b,j_history,p_hiatory ...w_final with w ..b_final with b

# plot cost versus iteration
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
# ax1.plot(J_hist[:100])
# ax2.plot(1000 + arange(len(J_hist[1000:])), J_hist[1000:])
# ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
# ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
# ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
# plt.show()