import tensorflow as tf
import numpy as np
from environment import Env
import gym
import threading
from a3c_network import *
import os

flags = tf.compat.v1.flags
flags.DEFINE_string('game', 'Breakout-v0', 'Name of the game to play.')
flags.DEFINE_integer('num_concurrent', 8, 'Number of concurrent actor-learner threads to use during training.')
flags.DEFINE_integer('tmax', 80000000, 'Number of training timesteps.')
flags.DEFINE_float('learning_rate', 0.0007, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_float('BETA', 0.01, 'factor of regularazation.')
flags.DEFINE_string('checkpoint_dir', '/tmp/checkpoints/', 'Directory for storing model checkpoints')
flags.DEFINE_boolean('show_training', True, 'If true, have gym render evironments during training')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_string('checkpoint_path', 'path/to/recent.ckpt', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_integer('num_eval_episodes', 100, 'Number of episodes to run gym evaluation.')
flags.DEFINE_integer('checkpoint_interval', 1000000, 'Checkpoint the model (i.e. save the parameters) every n ')
FLAGS = flags.FLAGS


def get_num_actions(game):
    env = gym.make(game)
    env = Env(env, 84, 84, 4)
    num_actions = len(env.gym_actions)
    return num_actions


def main():
    sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8))
    workers = []
    num_actions = get_num_actions(FLAGS.game)
    global_network = Worker(-1, num_actions, FLAGS.learning_rate, sess)
    
    for i in range(FLAGS.num_concurrent):
        workers.append(Worker(i, num_actions, FLAGS.learning_rate, sess, glob_net=global_network))
    saver = tf.train.Saver()

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not FLAGS.testing:
        sess.run(tf.global_variables_initializer())
        envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)]
        actor_learner_threads = [threading.Thread(target=workers[i].train, args=(envs[i], FLAGS.checkpoint_interval,
                                                                                 FLAGS.checkpoint_dir, saver)) for i in range(FLAGS.num_concurrent)]
        for t in actor_learner_threads:
            t.start()
        for t in actor_learner_threads:
            t.join()
    else:
        saver.restore(sess, FLAGS.checkpoint_path)
        env = gym.make(FLAGS.game)
        global_network.test(env)    

main()
