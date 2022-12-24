## Creating custom Reinforcement Learning Environments

The code file below is a raw driver code for a custom RL environment similar to a pre-built env.(s) like OpenAI Gym Environments
Creating a Environment class with all the neccessary methods including designing rewards and action policies etc.
The Gridworld problem stated below is solved using a simple Q-Learning epsilon-greedy approach which involves creation of Q-Table and performing DP Update rules via sampling the maximum/greedy actions for a given set of states.

### Q-Learning DP Update rule:

![[Pasted image 20221224192555.png]]

