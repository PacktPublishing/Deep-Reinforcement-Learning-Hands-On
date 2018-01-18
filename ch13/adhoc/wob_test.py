import gym
import universe


def configure(env):
    e = env
    while True:
        if e.metadata.get('configure.required') and hasattr(e, '_configure') and not isinstance(e, gym.Wrapper):
            e._configure()
            print("Configured!")
            break
        e = e.env

if __name__ == "__main__":
    env = gym.make("wob.mini.BisectAngle-v0")
    configure(env)
    print(env)
##    obs = env.reset()
#    print(obs)
    env.close()
    pass
