# Reinforcement Learning
:label:`chap_reinforcement_learning`

Reinforcement Learning (RL) is a suite of techniques that allows us to build machine learning systems that take decisions sequentially. For example, a package containing new clothes that you purchased from an online retailer arrives at your doorstep after a sequence of decisions, e.g., the retailer finding the clothes in the warehouse closest to your house, putting the clothes in a box, transporting the box by land or by air, and delivering it to your house within the city. There are many variables that affect the delivery of the package along the way, e.g., whether or not the clothes were available in the warehouse, how long it took to transport the box, whether it arrived in your city before the daily delivery truck left, etc. The key idea is that at each stage these variables that we do not often control affect the entire sequence of events in the future, e.g., if there were delays in packing the box in the warehouse the retailer may need to send the package via air instead of ground to ensure a timely delivery. Reinforcement Learning methods allow us to take the appropriate action at each stage of a sequential decision making problem in order to maximize some utility eventually, e.g., the timely delivery of the package to you.

Such sequential decision making problems are seen in numerous other places, e.g., while playing [Go](https://en.wikipedia.org/wiki/Go_(game)) your current move determines the next moves and the opponent's moves are the variables that you cannot control; a sequence of moves eventually determines whether or not you win. The movies that Netflix recommends to you now determine what you watch, whether you like the movie or not is unknown to Netflix, eventually a sequence of movie recommendations determines how satisfied you are with Netflix. Reinforcement learning is being used today to develop effective solutions to these problems :cite:`mnih2013playing,Silver.Huang.Maddison.ea.2016`. The key distinction between reinforcement learning and standard deep learning is that in standard deep learning the prediction of a trained model on one test datum does not affect the predictions on a future test datum; in reinforcement learning decisions at future instants (in RL, decisions are also called actions) are affected by what decisions were made in the past.

In this chapter, we will develop the fundamentals of reinforcement learning and obtain hands-on experience in implementing some popular reinforcement learning methods. We will first develop a concept called a Markov Decision Process (MDP) which allows us to think of such sequential decision making problems. An algorithm called Value Iteration will be our first insight into solving reinforcement learning problems under the assumption that we know how the environment behaves. In RL, the environment is the part of the problem that the robot does not control: how the state changes and which reward arrives in response to each action. Q-Learning, the sample-based counterpart of Value Iteration, will then let us take appropriate actions even when we do not necessarily have full knowledge of the environment. We will then study policy gradient methods, which learn the policy directly, instead of deriving it from a value function, by performing gradient ascent on the expected return, and reduce the variance of their updates with baselines. We then move these methods onto neural networks, develop actor-critic algorithms and Proximal Policy Optimization, and revisit Q-Learning at network scale with Deep Q-Networks. The chapter ends with the question of which data may drive which update, including the offline setting where the robot must learn from a fixed dataset without interacting at all. These techniques form the basis of more advanced RL algorithms that are used today in a variety of real-world applications, some of which we will point to in the chapter.

![Reinforcement Learning Structure](../img/RL_main.png)
:width:`400px`
:label:`fig_rl_big`

```toc
:maxdepth: 2

mdp
value-iter
qlearning
policy-gradient
baselines
deep-rl
actor-critic
ppo
dqn
offline-rl
```
