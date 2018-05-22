import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_data = X.shape[0]
  for i in range(0,num_data):
    scores = X[i].dot(W)
    scores = scores - np.max(scores)
    false_prob = np.sum(np.exp(scores))
    scores_p =np.exp(scores)/false_prob
        
    for j in range(0, num_classes):
        if j != y[i]:
            dW[:,j] += (np.exp(scores[j]) * X[i])/false_prob
        else:
            dW[:,j] += ((np.exp(scores[y[i]])*X[i])/false_prob) - X[i]
            
    
    loss += -scores[y[i]] + np.log(false_prob)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_data
  loss += reg * np.sum(W**2)
  dW /= num_data
  dW += 2*reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  num_data = X.shape[0]
  num_classes = W.shape[1]
  test = scores[np.arange(num_data),:]
  loss = np.log(np.sum(np.exp(scores[np.arange(num_data)]), axis = 1))-scores[np.arange(num_data), y]
  loss = np.sum(loss)
  
  loss /= num_data
  loss += reg * np.sum(W**2)  
  
  manipulated_scores = np.exp(scores)
  manipulated_scores /= np.sum(manipulated_scores, axis = 1, keepdims = True)
  #manipulated_scores[np.arange(num_data), y] -= 1
  dW = np.dot(X.T,manipulated_scores)
  
  for i in range(num_data):
      dW[:, y[i]] -= X[i].T
  
  dW /= num_data
  dW += 2 * reg*W
  #print(dW)
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

