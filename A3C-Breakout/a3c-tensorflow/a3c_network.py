import tensorflow as tf
import numpy as np
from environment import Env
import time
import scipy.signal
import random
# global step counter
T = 0
EPSILON = 1e-10


# initialization code from https://github.com/miyosuda/async_deep_reinforce

def fc_variable(weight_shape):
    input_channels = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
    return weight, bias


def conv_variable(weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
    return weight, bias


# sample policy code from https://github.com/coreylynch/async-rl
def sample_policy_action(probs):
    probs = probs - np.finfo(np.float32).epsneg
    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index
    

def initialize_weights(output_size, scope):
    weights = dict()
    scope.reuse_variables()
    weights["conv1_w"], weights["conv1_b"] = conv_variable([8, 8, 4, 16])
    
    weights["conv2_w"], weights["conv2_b"] = conv_variable([4, 4, 16, 32])
    
    weights["fc1_w"], weights["fc1_b"] = fc_variable([2592, 256])
    
    weights["plc_w"], weights["plc_b"] = fc_variable([256, output_size])
    
    weights["value_w"], weights["value_b"] = fc_variable([256, 1])
    
    return weights


class Worker:

    def __init__(self, thread_id, output_size, learning_rate, sess, beta=0.01, tmax=5, glob_net=None):
        self.thread_id = thread_id
        self.scope = "net" + str(thread_id)
        self.learning_rate = learning_rate
        # define placeholder for the input of the neural network
        self.input_state = tf.placeholder("float", [None, 84, 84, 4])

        self.sess = sess
        self.tmax = tmax
        self.output_size = output_size
        self.beta = beta
        self.TMAX = 80000000
        global T
        
        with tf.variable_scope(self.scope) as scope:
            self.weights = initialize_weights(self.output_size, scope)
            self._create_network(self.weights)
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        if glob_net != None:
            self.glob_net = glob_net
            self.sync_op = self._create_sync_operation(self.glob_net)
            
            self._create_loss()
            self._create_optimizer()

    def _create_network(self, weights):
        
        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input_state, weights["conv1_w"], strides=[1, 4, 4, 1], padding="VALID") +
                             weights["conv1_b"])
                                            
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, weights["conv2_w"], strides=[1, 2, 2, 1], padding="VALID") +
                             weights["conv2_b"])

        conv_output_flat = tf.reshape(h_conv2, [-1, 2592])
        
        h_fc1 = tf.nn.relu(tf.matmul(conv_output_flat, weights["fc1_w"]) + weights["fc1_b"])
        
        # define policy and state value output 
        self.policy = tf.nn.softmax(tf.matmul(h_fc1, weights["plc_w"]) + weights["plc_b"] + EPSILON)
        self.state_value = tf.matmul(h_fc1, weights["value_w"]) + weights["value_b"]
        
    def _create_loss(self):

        self.advantage = tf.placeholder("float", [None])
        self.targets = tf.placeholder("float", [None])
        self.actions = tf.placeholder("float", [None, self.output_size])
        
        log_pol = tf.log(self.policy + EPSILON)
        entropy = -tf.reduce_sum(self.policy * log_pol, reduction_indices=1)

        policy_loss = -tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pol, self.actions),
                                     reduction_indices=1) * self.advantage + self.beta * entropy)
        
        value_loss = 0.5 * tf.reduce_sum(tf.square(self.targets - self.state_value))

        self.loss = policy_loss + 0.5 * value_loss
            
    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(0.0001)
        self.grads = tf.gradients(self.loss, self.train_vars)
        # self.grads, _ = tf.clip_by_global_norm(self.grads, 40)

        self.grads_and_vars = list(zip(self.grads, self.glob_net.train_vars))
        self.opt = tf.group(self.optimizer.apply_gradients(self.grads_and_vars), tf.shape(self.input_state)[0])

    def _create_sync_operation(self, glob_net):

        sync = [self.train_vars[j].assign(self.glob_net.train_vars[j]) for j in range(len(self.train_vars))]
        return tf.group(*sync)
        
    def train(self, env, checkpoint_interval, checkpoint_dir, saver, gamma=0.99):
        global T
        self.saver = saver

        # initialize environment
        time.sleep(3 * self.thread_id)
        env = Env(env, 84, 84, 4)
        
        print('Starting thread ' + str(self.thread_id))

        terminal = False
        # Get initial game observation

        state = env.get_initial_state()
        
        # episode's reward and cost
        episode_reward = 0
        total_cost = 0
        counter = 0
        
        while T < self.TMAX:
            
            # lists for feeding placeholders
            states = []
            actions = []
            prev_reward = []
            state_values = []
            
            t = 0
            t_start = t
            self.sess.run(self.sync_op)
            while not (terminal or ((t - t_start) == self.tmax)):
                
                # forward pass of network. Get probability of all actions
                probs, v = self.sess.run((self.policy, self.state_value),
                                         feed_dict={self.input_state: [state]})
                                                            
                probs = probs[0]
                v = v[0][0]
                # print the outputs of the neural network fpr sanity chack
                if T % 2000 == 0:
                    print(probs)
                    print(v)

                # define list of actions. All values are zeros except , the
                # value of action that is executed
                action_list = np.zeros([self.output_size])

                # choose action based on policy
                action_index = sample_policy_action(probs)

                action_list[action_index] = 1

                # add state and action to list
                actions.append(action_list)
                states.append(state)
                
                state_values.append(v)
                
                # Gym executes action in game environment on behalf of actor-learner
                new_state, reward, terminal = env.step(action_index)

                # clip reward to -1, 1
                clipped_reward = np.clip(reward, -1, 1)
                prev_reward.append(clipped_reward)

                # Update the state and global counters
                state = new_state
                T += 1
                t += 1
                counter += 1
                
                # update episode's counter
                episode_reward += reward

                # Save model progress
                if T % checkpoint_interval < 200:
                    T += 200
                    self.saver.save(self.sess, checkpoint_dir+"/breakout.ckpt" , global_step = T)
                    
            if terminal:
                R_t = 0
            else:
                R_t = self.sess.run(self.state_value, feed_dict = {self.input_state : [state]})
                R_t = R_t[0][0]
            
            state_values.append(R_t)
            targets = np.zeros((t - t_start))

            for i in range(t - t_start - 1, -1, -1):
                R_t = prev_reward[i] + gamma * R_t
                targets[i] = R_t

            # compute the advantage based on GAE
            # code from https://github.com/openai/universe-starter-agent
            delta = np.array(prev_reward) + gamma * np.array(state_values[1:]) - np.array(state_values[:-1])
            advantage = scipy.signal.lfilter([1], [1, -gamma], delta[::-1], axis=0)[::-1]

            # update the global network
            cost, _ = self.sess.run((self.loss, self.opt), feed_dict={self.input_state: states,
                                                                      self.actions: actions,
                                                                      self.targets: targets,
                                                                      self.advantage: advantage})
            total_cost += cost
            
            if terminal:
                
                terminal = False
                print ("THREAD:", self.thread_id, "/ TIME", T, "/ REWARD", \
                    episode_reward, "/ COST", total_cost/counter)
                episode_reward = 0
                total_cost = 0
                counter = 0

                # Get initial game observation
                state = env.get_initial_state()

    def test(self, env):

        # initialize environment
        env = Env(env, 84, 84, 4)
        
        terminal = False
        # Get initial game observation
        state = env.get_initial_state()
        
        # episode's reward and cost
        episode_reward = 0
        
        for _ in range(100):
            while not terminal:
                
                # forward pass of network. Get probability of all actions
                probs, v = self.sess.run((self.policy, self.state_value),
                                                            feed_dict={self.input_state : [state]})
                                                            
                probs = probs[0]
                v = v[0][0]


                if random.random() < 0.01:
                    action_index = random.choice([0,1,2,3])
                else:
                    action_index = np.argmax(probs)

                # Gym excecutes action in game environment on behalf of actor-learner
                new_state, reward, terminal = env.step(action_index)
                env.env.render()
                # clip reward to -1, 1
                # Update the state and global counters
                state = new_state
                # update episode's counter
                episode_reward += reward
    
            if terminal:
                
                terminal = False
                print("THREAD:", self.thread_id, "/ TIME", T, "/ REWARD", \
                    episode_reward, "/ COST")
                episode_reward = 0
                counter = 0
                # Get initial game observation
                state = env.get_initial_state()

