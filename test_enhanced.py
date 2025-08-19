import gymnasium as gym
import BlackJackSim  # registers BlackjackSim-v0

env = gym.make("BlackjackSim-v0")

print("=== TEST 1: Basic game ===")
obs, info = env.reset()
env.render()
print(f"Initial: {obs}, info: {info}")

# Take action until done
action = 0  # STAY
obs, reward, terminated, truncated, info = env.step(action)
env.render()
print(f"After STAY: obs={obs}, reward={reward}, terminated={terminated}, info={info}")

print("\n=== TEST 2: Hit until bust or stand ===")
obs, info = env.reset()
env.render()
step_count = 0
while not (terminated or truncated) and step_count < 10:
    step_count += 1
    if obs[0] < 17:  # Hit if under 17
        action = 1  # HIT
        print(f"Step {step_count}: Taking HIT action")
    else:
        action = 0  # STAY
        print(f"Step {step_count}: Taking STAY action")
    
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"Step {step_count}: obs={obs}, reward={reward}, terminated={terminated}")
    
    if terminated or truncated:
        break

env.close()
