import gymnasium as gym
import BlackJackSim  # registers BlackjackSim-v0

print("🎰 TESTING BLACKJACK SCENARIOS 🎰\n")

for i in range(5):
    print(f"=== Game {i+1} ===")
    env = gym.make("BlackjackSim-v0")
    obs, info = env.reset()
    
    print(f"Initial state:")
    env.render()
    
    # If player has blackjack, it should be immediately detected
    if info['player_blackjack']:
        print("🔥 Player has blackjack - should be detected automatically!")
        obs, reward, terminated, truncated, info = env.step(0)  # Any action should trigger result
        env.render()
        print(f"Final reward: {reward}")
    else:
        # Play optimally: hit if under 17, stand if 17+
        while True:
            if obs[0] < 17:
                action = 1  # HIT
                print(f"Taking HIT (total: {obs[0]})")
            else:
                action = 0  # STAY 
                print(f"Taking STAY (total: {obs[0]})")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print("Game ended:")
                env.render()
                print(f"Final reward: {reward}")
                break
                
            if obs[0] > 21:
                print("💥 Player busted!")
                break
    
    env.close()
    print("\n" + "="*60 + "\n")

