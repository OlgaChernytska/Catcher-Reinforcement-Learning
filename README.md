# Playing Catcher game with Reinforcement Learning

Q-Learning                 Cross-Entropy Method               Neural Network       
![Q-Learning](https://github.com/OlgaChernytska/Catcher-Reinforcement-Learning/blob/master/figure/gif/catcher_q_learning.gif)       ![Cross-Entropy Method](https://github.com/OlgaChernytska/Catcher-Reinforcement-Learning/blob/master/figure/gif/catcher_cem.gif)       ![Neural Network](https://github.com/OlgaChernytska/Catcher-Reinforcement-Learning/blob/master/figure/gif/catcher_neural_network.gif)


## Motivation
Learn agent to play Catcher using Q-Learning, Cross-Entropy Method and Neural Network.

## Description

- Game Description

Agent has to catch fruits by moving left/right or doing nothing. If agent catches fruit, it receives reward of +1, when it loses, it receives negative reward of -1 and game ends. State is represented as numerical vector of length 4, containing agent x position, agent velocity, and x and y fruit positions.  

Metric to measure performance is number of timestamps (frames) that agent survives, because it is correlated with number of fruits caught and scores received. Performing random actions agent can survive for about 16 timestamp. I assume, that agent have learned to play game if he survives for 200 timestamps on average during last 200 episodes. 

- Q-Learning

This algorithm learns values for every possible state-action pair. All the values are stored in so called q-table, which has size of (number of discrete states) x (number of action). When playing, agent chooses actions based on q-table: action, that has the highest value among all the actions for the state is selected. 

Algorithm:
1) Discretize states.
2) Initialize Q-table contain zeros.
3) Loop until convergence: 
3.1) observe current state, select action, receive reward, observe next state; 
3.2) Update q-table using Bellman equation:

Q(s,a) := Q(s,a) + alpha * [r + gamma max_{a'} Q(s',a') - Q(s,a)]

Agent is encouraged to explore by introducing epsilon value - share of random actions. Epsilon decreases with episodes played.

Q-Learning is powerful and easy to implement. However, with the increase in dimensionality of state vector , number of discrete states increases rapidly and q-table becomes hard to learn. Additionally, there is no generalization - agent does not know what to do in the states he never observed and cannot infer action from observed other states. 

- Cross-Entropy Method

This method can be used for continuous state vector and there is no need in discretization. Goal is to find the best set of parameters - thetas, such that if agent uses the policy "If thetas.T * state > 0 , then Left, otherwise, Right" -- then he wins. Function F that evaluates the performance of the parameter vector is just number of timestamps survived - the more the better.

Algorithm:
1) Select batch size N and fraction of so called elite members - p.
2) For every parameter initialize mean to zero and variance to one.
3) Loop until convergence:
3.1) sample N from parameter distribution;
3.2) for all N parameter set evaluate its F (score);
3.3) select best p\*N parameter sets based on F (elite members);
3.4) update parameter mean and variance to be mean and variance of elite members.

Cross-Entropy Method is simple in implementation and very intuitive - try different parameter and select the best among them. In such implementation of policy function, algorithm works only for action set of two, but other policy functions can be used as well. However, training is highly unstable, algorithm has to be initialized several times until it finally starts to converge.

- Neural Network

Neural Networks show good result in all the Machine Learning fields and Reinforcement Learning is not an exception. Idea of the approach is similar to Q-Learning - learn value function and, when playing, choose actions with the highest value for the state. So trained network accepts as inputs state vector and action vector (concatenated) and returns value for this state-action pair. Action vector is one-hot encoded vector with length equal to dimensionality of action space.


Algorithm:
1) Create network architecture and initialize weights. Number of input neurones is equal to dimensionality of state vector plus dimensionality of action set. There is one output neuron with is q-value. This is regression problem, so mean squared error should be used as cost function.
2) Initialize memory array to store state, action, reward and next state, initialize its size - B. Network will be trained using batches of size B.
3) Loop until convergence:
2.1) Play B states until memory array is full. Store state, action, reward and next state in the memory array. 
2.2) Make sure that state vector entries are normalized.
2.3) Create x and y vectors for every observation. Vector x is concatenated inputs state vector (normalized) and one-hot encoded action vector. Vector y has to be generated from network with current weights using Bellman equation: 
y = Q(s,a)  = r + gamma * max_{a'} Q(s',a').
3.4) Train network for all samples in the batch.
3.5) Empty memory array.
Epsilon trick is used here as well.

Neural Networks can deal with any kind of game complexity, however, a lot of computational power is needed to train them.

## Deliverables

- [x] notebooks - 3 notebooks for learning model
- [x] logs - training logs
- [x] models - folder to save learned parameter
- [x] demo - demonstration how agent learns to play


## Interpretation

There is a lot
Learning is ustable. It is highly depends what game states agent see at the beginning of the training. So all the algorithm have to be reinitialize several times not to just achieve good results, but launch training process at all.
