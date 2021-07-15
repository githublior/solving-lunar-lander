# Solving-lunar-lander


Project solving the LunarLanderContinuous-v2 from the openai-gym (https://gym.openai.com/envs/LunarLanderContinuous-v2/) , using techniques learnt in course Reinforcement learning.

the goal :
land the space-ship between the flags smoothly.
The ship has 3 throttles in it. One throttle points downward and the other 2 points in the
left and right direction. With the help of these, you have to control the Ship.
Observation Space: [Position X, Position Y, Velocity X, Velocity Y, Angle, Angular
Velocity, Is left leg touching the ground: 0 OR 1, Is right leg touching the ground: 0 OR 1]
Continuous Action Space: Two floats [main engine, left-right engines].
Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with
less than 50% power.
Left-right: -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
Please note that there are 2 different Lunar Lander Environments in OpenAIGym. One
has discrete action space and the other has continuous action space.
we will solve here the continuous one.

To output discrete action space we will have to quantize the action into a finite number
of states.we will use atleast X2 states than in the discrete case.
 ****Solving the LunarLanderContinuous-v2 means getting an average reward of 200 over
100 consecutive trials.****

Link that demonstrate how to use/render the game (with just random actions):
https://colab.research.google.com/drive/1R5BwSTau9zuEj8r4Yh6gB3Nn7NXOm
-Fx?usp=sharing


Our goal:
1) To solve the environment
2) As fast as possible (we want a small number of crashes until learning the task) 
3) With a comparison/referring between different variants of what we have learnt in the
course (e.g. DQN, target network, network architecture, double-DQN, dueling DDQN,
experience replay, prioritized experience replay, TD(lambda), discount factor effects,
epsilon-greedy, tabular methods, quantizing effects., use terms from the course)
Reading more papers, Using some advanced policy gradients and actor-critic methods.

The goal of this part is to show our knowledge and theoretical/practical understanding of
the different algorithms and hyperparameters.

4) Then, refer to the Lunar-lander with uncertainty. (explanation: observations in the real
physical world are sometimes noisy). Specifically, we will add a zero-mean
Gaussian noise with mean=0 and std = 0.05 to PositionX and PositionY observation of
the location of the lander.
