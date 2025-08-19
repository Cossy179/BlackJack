import gymnasium as gym
import BlackJackSim

env = gym.make("BlackjackSim-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(1)  # HIT
env.render()  # Beautiful display with emojis