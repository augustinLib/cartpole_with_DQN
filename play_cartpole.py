import gym
import numpy as np
import tensorflow as tf
from train import Agent


if __name__=="__main__":

    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.load_weights('')
    time = 0
    state = env.reset()

    while True:
        env.render()
        qs = agent.actual_network(tf.convert_to_tensor([state], dtype=tf.float32))
        action = np.argmax(qs.numpy())

        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()