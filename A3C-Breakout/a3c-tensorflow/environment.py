import tensorflow as tf
from skimage.transform import resize
import skimage.color
import numpy as np
from collections import deque
import random
from PIL import Image
import cv2
import time
#from scipy.misc import imresize


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# code is based on https://github.com/coreylynch/async-rl
class Env(object):

    def __init__(self, gym_env, resized_width, resized_height, agent_history_length):
        self.env = gym_env
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
      
        if gym_env.spec.id == "PongDeterministic-v3" or gym_env.spec.id == "BreakoutDeterministic-v3":
            self.gym_actions = [0, 1, 2, 3]
        else:
            self.gym_actions = range(self.env.action_space.n)
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer
        """
        # Clear the state buffer
        self.state_buffer = deque()
    
        x_t = self.env.reset()
        for _ in range(10):
            x_t, r_t, terminal, info = self.env.step(0)
            
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 0)
        
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)

        s_t = np.transpose(s_t, (1, 2, 0))
        return s_t

    def get_preprocessed_frame(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """
        grayscale_observation = rgb2gray(observation).astype('float32')
        grayscale_observation = cv2.resize(grayscale_observation, size=(110, 84, 1))
        grayscale_observation = grayscale_observation[18:102, :]

        x_t = grayscale_observation / 255.0

        # uncomment to test preprocessing
        # a = x_t * 255.0
        # im = Image.fromarray(a)
        # im.show()
        # time.sleep(1)
        return x_t

    def step(self, action_index):
        """
        Executes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """

        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
        s_t1[:self.agent_history_length-1, ...] = previous_frames
        s_t1[self.agent_history_length-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)
        s_t1 = np.transpose(s_t1, (1,2,0))

        return s_t1, r_t, terminal
