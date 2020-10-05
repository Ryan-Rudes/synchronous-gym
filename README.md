# Synchronus Gym
A wrapper for OpenAI gym which enables use of multiple environments in synchrony

# Example
To use the wrapper, simply create a gym environment as normal, than reassign the variable storing this environment to the MultiGymWrapper object as follows:
```
import gym
from wrapper import MultiGymWrapper

# Make a standard gym environment
env = gym.make("Qbert-v0")

# Wrap the environment inside the multi-agent wrapper object
# The parameter n specifies the number of simultaneous simulations
env = MultiGymWrapper(env, n = 8)

# Run a random episode on all 8 environments simultaneously
states = env.reset()

while True:
  actions = env.action_space.sample()
  states, rewards, terminals, infos = env.step(actions)
  env.render()
  if any(terminals):
    break
    
# Close all 8 open simulations
env.close()
```
