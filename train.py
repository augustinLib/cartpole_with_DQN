import numpy as np
import matplotlib.pyplot as plt
from buffer import ReplayBuffer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# Q network
class DQN(Model):

    def __init__(self, action_n):
        super(DQN, self).__init__()

        self.layer1 = Dense(128, activation='relu')
        self.layer2 = Dense(64, activation='relu')
        self.layer3 = Dense(32, activation='relu')
        self.outputLayer = Dense(action_n, activation='linear')


    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        q = self.outputLayer(x)
        return q


class Agent(object):

    def __init__(self, env):

        ## 하이퍼파라미터
        self.discount_factor = 0.9
        self.batch_size = 64
        self.buffer_size = 20000
        self.learning_rate = 0.001
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.env = env

        
        self.state_dim = env.observation_space.shape[0]  # cart Positon, cart Velocity, Pole Angle, Pole Angular Velocity
        self.action_n = env.action_space.n   #push right, push left

        
        self.actual_network = DQN(self.action_n)
        self.target_network = DQN(self.action_n)

        self.actual_network.build(input_shape=(None, self.state_dim))
        self.target_network.build(input_shape=(None, self.state_dim))

        self.actual_network.summary()

        self.optimizer = Adam(self.learning_rate)
        ## experience replay 초기화
        self.buffer = ReplayBuffer(self.buffer_size)
        # 에피소드의 reward 저장하는 곳
        self.episode_reward_list = []


    def epsilon_greedy_policy(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()  #epsilon보다 작은 확률일 경우, explore
        else:
            qs = self.actual_network(tf.convert_to_tensor([state], dtype=tf.float32)) 
            return np.argmax(qs.numpy()) # 그렇지 않을 경우, exploit


    ## actual network를 target network로
    def update_target_network(self):
        phi = self.actual_network.get_weights()
        target_phi = self.target_network.get_weights()
        for i in range(len(phi)):
            target_phi[i] = phi[i] 
        self.target_network.set_weights(target_phi)


    # gradient 업데이트
    def learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, self.action_n)
            q = self.actual_network(states, training=True)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(td_targets - q_values)) #loss

        grads = tape.gradient(loss, self.actual_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actual_network.trainable_variables))


    # y_k = r_k + discount_factor* max Q(s_k+1, a)
    def target_value(self, rewards, target_qs, dones):
        max_q = np.max(target_qs, axis=1, keepdims=True)
        y_k = np.zeros(max_q.shape)
        for i in range(max_q.shape[0]): 
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.discount_factor * max_q[i]
        return y_k

    def load_weights(self, path):
        self.actual_network.load_weights(path + 'cartpole_dqn.h5')



    def train(self, max_episode_num):
        self.update_target_network()
        for ep in range(int(max_episode_num)):
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                action = self.epsilon_greedy_policy(state)
                next_state, reward, done, _ = self.env.step(action)
                train_reward = reward + time*0.01

                # replay buffer에 transition 추가
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000: 

                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay

                    # replay buffer에서 randomly하게 batch를 sample
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)

                    # target network에서 target Q-values 산출
                    target_qs = self.target_network(tf.convert_to_tensor(next_states, dtype=tf.float32))

                    y_i = self.target_value(rewards, target_qs.numpy(), dones)

                    self.learn(tf.convert_to_tensor(states, dtype=tf.float32), actions, tf.convert_to_tensor(y_i, dtype=tf.float32))
                    self.update_target_network()

                state = next_state
                episode_reward += reward
                time += 1


            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.episode_reward_list.append(episode_reward)

            self.actual_network.save_weights("cartpole_dqn.h5")
        np.savetxt('cartpole_epi_reward.txt', self.episode_reward_list)




    def plot_result(self):
        plt.plot(self.episode_reward_list)
        plt.show()