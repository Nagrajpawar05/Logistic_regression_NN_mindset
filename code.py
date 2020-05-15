#reading h5 file
import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('drive/My Drive/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('drive/My Drive/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
 
m_train=train_set_x_orig.shape[0] #num of training examples
m_test=test_set_x_orig.shape[0] #no. of test exapmples
num_px=train_set_x_orig.shape[1] #width of training ex (width of image )

print("Num of training examples : ",m_train)
print("Num of test examples : ",m_test)
print("Num of training labels : ",num_px)
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("shape of train__set",train_set_x_orig.shape)
print("shape of train_Y_set",train_set_y.shape)
print("shape of test__set",test_set_x_orig.shape)

#Now reshape the training examples
train_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T #create a matrix of nx(64x64x3) X m
test_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print ("train_set_x_flatten shape: " + str(train_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x = train_x_flatten/255. #to standardize the pixel values
test_set_x = test_x_flatten/255.

# GRADED FUNCTION: sigmoid

def sigmoid(z):


    s =1/(1+np.exp(-z))

    
    return s
    
 #  initialize_with_zeros
def initialize(dim):
 
    
  
    w = np.random.randn(dim,1)*0.1
    b = 0
  

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
    
def propagate(w, b, X, Y):

    
    m = X.shape[1]
    
 
    A = sigmoid(np.dot(w.T,X)+b)                                   # compute activation
    cost =(-1/m) *(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))                              # compute cost
  
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
   
    dw =(1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*(np.sum(A-Y))
 

    assert(dw.shape == w.shape)  #assert=>to make sure of dimensions are correct
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
    
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
     
        grads, cost =propagate(w,b,X,Y)
      
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
       
        w =w-learning_rate*dw
        b =b-learning_rate*db
      
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
    
def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture

    A = sigmoid(np.dot(w.T,X)+b)
    
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        
        Y_prediction[0,i]=1 if A[0,i]>0.5 else 0

    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
    
    
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

   # d -- dictionary containing information about the model.
    
    
   
    
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent 
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples 
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

  

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
    
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
