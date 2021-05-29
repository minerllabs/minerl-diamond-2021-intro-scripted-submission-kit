import random
import numpy as np
import gym
import torch as th
from torch import nn

MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamond.
TREECHOP_STEPS = 2000  # number of steps to run BC lumberjack for in evaluations.
N_WOOD_THRESHOLD = 4  # number of wood logs to get before starting script #2.


# !!! Do not change this! This is part of the submission kit !!!
class EpisodeDone(Exception):
    pass


# !!! Do not change this! This is part of the submission kit !!!
class Episode(gym.Env):
    """A class for a single episode."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i


def str_to_act(env, actions):
    """
    Simplifies specifying actions for the scripted part of the agent.
    Some examples for a string with a single action:
        'craft:planks'
        'camera:[10,0]'
        'attack'
        'jump'
        ''
    There should be no spaces in single actions, as we use spaces to separate actions with multiple "buttons" pressed:
        'attack sprint forward'
        'forward camera:[0,10]'
    :param env: base MineRL environment.
    :param actions: string of actions.
    :return: dict action, compatible with the base MineRL environment.
    """
    act = env.action_space.noop()
    for action in actions.split():
        if ":" in action:
            k, v = action.split(':')
            if k == 'camera':
                act[k] = eval(v)
            else:
                act[k] = v
        else:
            act[action] = 1
    return act


def get_action_sequence_bulldozer():
    """
    Specify the action sequence for Bulldozer, the scripted lumberjack.
    """
    action_sequence_bulldozer = []
    action_sequence_bulldozer += [''] * 100  # wait 5 secs
    action_sequence_bulldozer += ['camera:[10,0]'] * 3  # look down 30 degrees

    for _ in range(100):
        action_sequence_bulldozer += ['attack sprint forward'] * 100  # dig forward for 5 secs
        action_sequence_bulldozer += ['jump']  # jump!
        action_sequence_bulldozer += ['attack sprint forward'] * 100
        action_sequence_bulldozer += ['jump']
        action_sequence_bulldozer += ['attack sprint forward'] * 100
        if random.random() < 0.5:  # turn either 90 degrees left or 90 degrees right with an equal probability
            action_sequence_bulldozer += ['camera:[0,-10]'] * 9
        else:
            action_sequence_bulldozer += ['camera:[0,10]'] * 9
    return action_sequence_bulldozer


def get_action_sequence():
    """
    Specify the action sequence for the scripted part of the agent.
    """
    # make planks, sticks, crafting table and wooden pickaxe:
    action_sequence = []
    action_sequence += [''] * 100
    action_sequence += ['craft:planks'] * 4
    action_sequence += ['craft:stick'] * 2
    action_sequence += ['craft:crafting_table']
    action_sequence += ['camera:[10,0]'] * 18
    action_sequence += ['attack'] * 20
    action_sequence += [''] * 10
    action_sequence += ['jump']
    action_sequence += [''] * 5
    action_sequence += ['place:crafting_table']
    action_sequence += [''] * 10

    # bug: looking straight down at a crafting table doesn't let you craft. So we look up a bit before crafting.
    action_sequence += ['camera:[-1,0]']
    action_sequence += ['nearbyCraft:wooden_pickaxe']
    action_sequence += ['camera:[1,0]']
    action_sequence += [''] * 10
    action_sequence += ['equip:wooden_pickaxe']
    action_sequence += [''] * 10

    # dig down:
    action_sequence += ['attack'] * 600
    action_sequence += [''] * 10

    return action_sequence


class MineRLAgent():
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.

    NOTE:
        This class enables the evaluator to run your agent in parallel in Threads,
        which means anything loaded in load_agent will be shared among parallel
        agents. Take care when tracking e.g. hidden state (this should go to run_agent_on_episode).
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        pass

    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        NOTE:
            This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        env = single_episode_env

        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        action_sequence_bulldozer = get_action_sequence_bulldozer()
        action_sequence = get_action_sequence()

        # scripted part to get some logs:
        for j, action in enumerate(action_sequence_bulldozer[:MAX_TEST_EPISODE_LEN]):
            obs, reward, done, _ = env.step(str_to_act(env, action))
            total_reward += reward
            steps += 1
            if obs['inventory']['log'] >= N_WOOD_THRESHOLD:
                break
            if done:
                break

        # scripted part to use the logs:
        if not done:
            for i, action in enumerate(action_sequence[:MAX_TEST_EPISODE_LEN - j]):
                obs, reward, done, _ = env.step(str_to_act(env, action))
                total_reward += reward
                steps += 1
                if done:
                    break
