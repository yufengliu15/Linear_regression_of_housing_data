import copy, math
import numpy as np
import matplotlib.pyplot as plt
import csv
#plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# =========================== RETRIEVING DATA =====================
with open ('data.csv', 'r') as dataFile: 
    # setting up the data
    filereader = csv.reader(dataFile)
    features = []
    pricing = []
    firstProperty = True
    for property in filereader:
        tempFeature = []
        # skips over data w/o square footage, properties that have multiple options, and the first line
        if (property[3] != "N/A" and ("-" not in property[4] and "-" not in property[3]) and not(firstProperty)):
            # bedrooms
            tempFeature.append(float(property[1]))
            # bathrooms
            tempFeature.append(float(property[2]))
            # square footage
            tempFeature.append(int(property[3]))
            # pricing
            pricing.append(int(property[4][1:]))
            
            features.append(tempFeature)
        firstProperty = False
        
# =========================== FUNCTIONS =============================
def predict_single_loop(x, w, b):
    """
    single predict using linear regression

    Args:
        x (ndarray): example with multiple features
        w (ndarray): model parameters
        b (scalar): model parameter
    Returns:
        p (scalar): prediction
    """
    p = np.dot(x, w) + b 
    return p    

def compute_cost(X,y,w,b):
    """
    compute cost

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray): target values
        w (ndarray): model parameters
        b (scalar): model parameter
    Returns:
        cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = predict_single_loop(X[i], w, b)
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return cost

def compute_gradient(X,y,w,b):
    """
    computes the gradient for linear regression 

    Args:
        X (ndarray): Data, m examples with n features
        y (ndarray): target values 
        w (ndarray): model parameters 
        b (scalar): model parameter
    Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameter w.
        dj_db (scalar):  The gradient of the cost w.r.t. the parameter b.
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db += err
    dj_dw = dj_dw / m 
    dj_db = dj_db / m
    
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        w_in (_type_): _description_
        b_in (_type_): _description_
        cost_function (_type_): _description_
        gradient_function (_type_): _description_
        alpha (_type_): _description_
        num_iters (_type_): _description_
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in) # avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X,y,w,b)
        
        # update parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # save cost j at each iteration
        if i < 100000:
            J_history.append(cost_function(X,y,w,b))
        
        # print cost at every tenth of num_iters
        if i % math.ceil(num_iters / 10) == 0:
            print (f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")
    return w, b, J_history

def zscore_normalize_features(X):
    """
    computes X, zcore normalized by column

    Args:
        X (ndarray (m,n)): input data, m examples, n features
    
    Returns:
        X_norm (ndarray (m,n)): input normalized by column
        mu (ndarray (n,)): mean of each feature
        sigma (ndarray (n,)): standard deviation of each feature
    """
    
    # find the mean of each column/feature
    mu = np.mean(X, axis=0) # mu will have shape (n,)
    # find the standard deviation of each column/feature 
    sigma = np.std(X, axis=0) # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu)/ sigma
    
    return (X_norm, mu, sigma)
    
def run_gradient_descent(X_train, y_train, iterations, alpha):
    # initialize parameters
    initial_w = np.zeros_like(w_init)
    initial_b = 0.

    # run gradient descent 
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

    # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
    plt.show()
    
    return w_final, b_final, J_hist

# initializing variables for ML
X_train = np.array(features)
X_features = ["bedrooms", "bathrooms", "size (sq ft)"]
y_train = np.array(pricing)

# random values for w & b
b_init = 585.1811367994083
w_init = np.array([3.9133535, 1.75376741, 0.36032453])

# graphs for each feature 
fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price ($/month)")
plt.show()

# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 300, 1.0e-1)

#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,3,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=["orange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()