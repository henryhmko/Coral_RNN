import numpy as np
import random
import copy

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def clip(gradients, maxValue):

  # Clips the gradients' values between minimum and maximum.

  gradients = copy.deepcopy(gradients)

  dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']


  for gradient in gradients:
    np.clip(gradients[gradient], -maxValue, maxValue, out = gradients[gradient])

  gradients = {'dWaa': dWaa, 'dWax': dWax, 'dWya': dWya, 'db': db, 'dby':dby}

  return gradients


def sample(parameters, char_to_ix):

  # Sample a sequence of characters according to a sequence of probability distributions output of the RNN


  # Retrieve parameters and relevant shapes from "parameters" dictionary
  Waa, Wax, Wya, by, b = parameters["Waa"], parameters["Wax"], parameters["Wya"], parameters["by"], parameters["b"]
  vocab_size = by.shape[0]
  n_a = Waa.shape[1]

  # Create a zero vector x than can used as the one-hot vector
  x = np.zeros((vocab_size, 1))
  a_prev = np.zeros((n_a, 1))

  indices = []

  # idx is the index of the one-hot vector x that is set to 1
  # initialize idx to -1
  idx = -1

  #Loop over t time-steps. At each time-step...:
  # Sample a character from a probability distribution
  # Append its index('idx') to the list "indices".
  # Stop when it reaches 50 characters(highly unlikely)
  # Setting max character number helps with debugging and prevents infinite loops.

  counter = 0
  newline_character = char_to_ix['\n']

  while (idx != newline_character and counter != 50):

    # Forward prop x

    a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
    z = np.dot(Wya, a) + by
    y = softmax(z)

    # Sample the index of a character within the vocabulary from the probability distribution y
    idx = np.random.choice(range(len(y)), p = np.squeeze(y))

    # Append the index to "indices"
    indices.append(idx)

    # Overwrite the input x with one that corresponds to the sampled index 'idx'
    x = np.zeros((vocab_size, 1))
    x[idx] = 1

    # Update a_prev to a
    a_prev = a

    counter += 1

  if counter == 50:
    indices.append(char_to_ix['\n'])
  
  return indices