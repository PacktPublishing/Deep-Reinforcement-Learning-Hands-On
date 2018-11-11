# Deep Reinforcement Learning Hands-On

Code samples for [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/big-data-and-business-intelligence/practical-deep-reinforcement-learning)
book

## Versions and compatibility

This repository is being maintained by book author [Max Lapan](https://github.com/Shmuma).
I'm trying to keep all the examples working under the latest versions of [PyTorch](https://pytorch.org/) 
and [gym](https://gym.openai.com/), which is not always simple, as software evolves. For example, OpenAI Universe, 
extensively being used in chapter 13, was discontinued by OpenAI. List of current requirements is present in 
[requirements.txt](requirements.txt) file.

And, of course, bugs in examples are inevitable, so, exact code might differ from code present in the book text.

Too keep track of major code change, I'm using tags and branches, for example:
* [tag 01_release](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/01_release) marks code 
state right after book publication in June 2018
* [branch master](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On) has the latest 
version of code updated for the latest stable PyTorch 0.4.1
* [branch torch_1.0](not_created_yet) keeps the activity of porting examples to PyTorch 1.0 (not yet released)

## Chapters' examples

* [Chapter 2: OpenAI Gym](Chapter02)
* [Chapter 3: Deep Learning with PyTorch](Chapter03)
* [Chapter 4: Cross Entropy method](Chapter04)
* [Chapter 5: Tabular learning and the Bellman equation](Chapter05)
* [Chapter 6: Deep Q-Networks](Chapter06)
* [Chapter 7: DQN extensions](Chapter07)
* [Chapter 8: Stocks trading using RL](Chapter08)
* [Chapter 9: Policy Gradients: an alternative](Chapter09)
* [Chapter 10: Actor-Critic method](Chapter10)
* [Chapter 11: Asynchronous Advantage Actor-Critic](Chapter11)
* [Chapter 12: Chatbots traning with RL](Chapter12)
* [Chapter 13: Web navigation](Chapter13)
* [Chapter 14: Continuous action space](Chapter14)
* [Chapter 15: Trust regions: TRPO, PPO and ACKTR](Chapter15)
* [Chapter 16: Black-box optimisation in RL](Chapter16)
* [Chapter 17: Beyond model-free: imagination](Chapter17)
* [Chapter 18: AlphaGo Zero](Chapter18)


# Deep Reinforcement Learning Hands-On
This is the code repository for [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/big-data-and-business-intelligence/deep-reinforcement-learning-hands?utm_source=github&utm_medium=repository&utm_campaign=9781788834247), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
Recent developments in reinforcement learning (RL), combined with deep learning (DL), have seen unprecedented progress made towards training agents to solve complex problems in a human-like way. Google’s use of algorithms to play and defeat the well-known Atari arcade games has propelled the field to prominence, and researchers are generating new ideas at a rapid pace.

Deep Reinforcement Learning Hands-On is a comprehensive guide to the very latest DL tools and their limitations. You will evaluate methods including Cross-entropy and policy gradients, before applying them to real-world environments. Take on both the Atari set of virtual games and family favorites such as Connect4. The book provides an introduction to the basics of RL, giving you the know-how to code intelligent learning agents to take on a formidable array of practical tasks. Discover how to implement Q-learning on ‘grid world’ environments, teach your agent to buy and trade stocks, and find out how natural language models are driving the boom in chatbots.
## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.
The code will look like the following:
```
def get_actions(self):
 return [0, 1]
```
## Related Products
* [Deep Learning with TensorFlow - Second Edition](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-tensorflow-second-edition?utm_source=github&utm_medium=repository&utm_campaign=9781788831109)

* [Python Interviews](https://www.packtpub.com/web-development/python-interviews?utm_source=github&utm_medium=repository&utm_campaign=9781788399081)

* [Python Machine Learning - Second Edition](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-second-edition?utm_source=github&utm_medium=repository&utm_campaign=9781787125933)
