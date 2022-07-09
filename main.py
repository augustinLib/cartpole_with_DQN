import tensorflow as tf
from train import Agent
import gym


if __name__=="__main__":
    max_episode_num = 500
    env = gym.make("CartPole-v1")
    agent = Agent(env)

    with tf.device("/device:GPU:0"):
        agent.train(max_episode_num)

    agent.plot_result()




    