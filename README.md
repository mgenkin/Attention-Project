# Attention-Project

#### First attention model posted

##### Code

The non-learning `first_attention_model.py` file depends on pygame and numpy.

The neural net `first_attention_model_nn.py` file depends on pygame, numpy, theano, lasagne.

The agent's "sight" is modeled as follows:

The 180-degree angle around the direction the agent is facing is divided into bins.

For each bin, we receive an input value that corresponds to the inverse distance (closeness) of the closest obstacle in that direction.

In the non-learning model, the agent simply chooses a direction to move in that's at right angles to the direction of the closest obstacle.

In the learning model, the agent incurs a cost that corresponds to the closeness to the closest obstacle, and a neural network is trained to predict that cost.  The input to the network is the agent's vision (`views`) and the direction (encoded as a one-hot vector), concatenated into one vector.  Then the model is used to predict possible cost for each direction choice, and the direction with minimum predicted cost is chosen.

##### =)

There are a lot of comments in the code, but please ask me if there's anything you don't understand; I don't usually share code, so there will definitely be some things I haven't explained properly.

After you have an idea of what's going on, please talk to me about plans for what to do next, I'm sure we both have plenty of ideas =)
