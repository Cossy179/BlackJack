try:
    from gymnasium.envs.registration import register
    # Expose the Gymnasium environment and register it so users can do gymnasium.make("BlackjackSim-v0")
    register(
        id="BlackjackSim-v0",
        entry_point="BlackJackSim.gym_env:BlackjackSimEnv",
    )
    __all__ = ["BlackjackSimEnv"]
except Exception:
    # Gymnasium is optional at import time; registration will be skipped if unavailable
    __all__ = []

