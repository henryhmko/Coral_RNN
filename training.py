from UTILS import *
from gradient_clipping import *

###### TRAINING CONFIG ######
learning_rate = 0.05
sample_per_iter = 1000   # Sample every <sample_per_iter> iterations
coral_samples = 15 # Generate <coral_sample> number of samples
num_iterations = 35000


####### Data Cleaning #######
data = open('corals_species.txt', 'r').read()
data = data.lower()
data = data.replace(" ", "_")

chars = list(set(data))
vocab_size = len(chars) #number of unique characters in data

chars = sorted(chars)
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}



###### TRAINING & SAMPLING ######

def model(data_x, ix_to_char, char_to_ix, num_iterations = num_iterations, n_a = 50, coral_names = coral_samples, vocab_size = vocab_size, verbose = False):

    # Trains the model and generates coral names. 


    # Retrive n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    #Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # Initialize loss
    loss = get_initial_loss(vocab_size, coral_names)

    # Built list of all coral names (training examples)
    examples = [x.strip() for x in data_x]

    # Shuffle list of all corals names
    np.random.shuffle(examples)

    # Initialize the hidden state of  LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization Loop
    for j in range(num_iterations):
      
      # Set the index 'idx'
      idx = j % len(examples)

      # Set the input X
      single_example_chars = examples[idx]
      single_example_ix = [char_to_ix[c] for c in single_example_chars]

      # if X[t] == None, we just have x[t]=0
      X = [None] + single_example_ix
      
      # Set the labels Y
      Y = X[1:]
      ix_newline = [char_to_ix["\n"]]
      Y = Y + ix_newline

      # Perform one optimization step: Forwardprop -> Backprop -> Clip -> Update params
      curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = learning_rate)

      if verbose and j in [0, len(examples) -1, len(examples)]:
        print("j = ", j, "idx = ", idx,)
      if verbose and j in [0]:
        print("single_example_chars", single_example_chars)
        print("single_example_ix", single_example_ix)
        print(" X = ", X, "\n", "Y =       ", Y, "\n")

      # Use a latency trick to keep the loss smooth. Accelerates training
      loss = smooth(loss, curr_loss)

      #Generate 'n' characters to check model outputs every [sample_per_iter] iterations
      if j % sample_per_iter == 0:
        print("Iteration: %d, Loss: %f" % (j, loss) + '\n')

        #number of coral names to print
        for name in range(coral_names):

          # Sample indices and print them
          sampled_indices = sample(parameters, char_to_ix)
          last_coral_name = get_sample(sampled_indices, ix_to_char)
          print(last_coral_name.replace('\n', ''))

        print('\n')
      
    return parameters, last_coral_name
    



## RUN FOR TRAINING & SAMPLING ##
## Uncomment line below
#parameters, _ = model(data.split('\n'), ix_to_char, char_to_ix, num_iterations, coral_names = coral_samples, verbose=True)