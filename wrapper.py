import gym

class InvalidArgumentError(Exception):
    pass

class MultiGymWrapper:
    def __init__(self, env, n):
        self.envs = [gym.make(env.spec.id) for i in range(n)]
        self.n = n

        self.action_space = env.action_space
        self.class_name = env.class_name
        self.metadata = env.metadata
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.spec = env.spec
        self.unwrapped = env.unwrapped

        self.get_action_meanings = env.env.get_action_meanings
        self.get_keys_to_action = env.env.get_keys_to_action
        self.np_random = [env.env.np_random for env in self.envs]
        self.game_path = env.env.game_path
        self.game_mode = env.env.game_mode
        self.viewer = env.env.viewer
        self.game_difficulty = env.env.game_difficulty
        self.game = env.env.game
        self.frameskip = env.env.frameskip

        self.action_space.sample = lambda: [env.action_space.sample() for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()

    def compute_reward(self, achieved_goal, desired_goal, info):
        return [env.compute_reward(achieved, desired, i) for env, achieved, desired, i in zip(self.envs, achieved_goal, desired_goal, info)]

    def render(self, mode='human', which='one'):
        if mode == 'rgb_array':
            if which == 'one':
                return self.envs[0].render(mode = 'rgb_array')
            elif which == 'all':
                return [env.render(mode = 'rgb_array') for env in self.envs]
            else:
                raise InvalidArgumentError("Invalid argument for parameter 'which'. Valid specifications are 'one' and 'all'.")
        elif mode == 'human':
            if which == 'one':
                self.envs[0].render(mode = 'human')
            elif which == 'all':
                for env in self.envs:
                    env.render(mode = 'human')
            else:
                raise InvalidArgumentError("Invalid argument for parameter 'which'. Valid specifications are 'one' and 'all'.")
        else:
            raise InvalidArgumentError("Invalid argument for parameter 'mode'. Valid specifications are 'rgb_array' and 'human'.")

    def reset(self):
        return [env.reset() for env in self.envs]

    def seed(self, seed=None):
        seeds = []
        for env in self.envs:
            seeds.append(env.seed(seed))
        return seeds

    def step(self, actions):
        states, rewards, terminals, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            state, reward, terminal, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            terminals.append(terminal)
            infos.append(info)
        return states, rewards, terminals, infos

    def clone_full_states(self):
        return [env.env.clone_full_state() for env in self.envs]

    def clone_states(self):
        return [env.env.clone_state() for env in self.envs]

    def restore_full_states(self, states):
        for state, env in zip(states, self.envs):
            env.env.restore_full_state(state)

    def restore_states(self, states):
        for state, env in zip(states, self.envs):
            env.env.restore_state(state)
