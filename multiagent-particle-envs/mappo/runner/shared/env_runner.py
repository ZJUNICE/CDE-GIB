
import time
import numpy as np
import torch
from mappo.runner.shared.base_runner import Runner
import imageio
import train

def _t2n(x):
    return x.detach().cpu().numpy() if type(x) != np.ndarray else x

class EnvRunner(Runner):

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        step_all = 0
        step_all1 = 0
        inference_time = 0
        for episode in range(episodes):

            t0 = time.time()
            x = 0
            y = 0
            self.warmup()
            if self._use_attn:
                self.trainer.policy.actor.init_att()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):

                values, actions, action_log_probs, rnn_states, rnn_states_critic,\
                    actions_env, hid_state, trigger_door_count = self.collect(step)


                obs, rewards, dones, infos = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, values, actions, action_log_probs,\
                    rnn_states, rnn_states_critic, hid_state

                x = x + self.trainer.policy.actor.data_size
                self.insert(data)


            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()

                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                print('value: ', train_infos['value_loss'], ' policy:', train_infos['policy_loss'])
                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            cost = time.time() - t0
            print("episode {}, step {}, cost time {:.3f}, inference time {:.3f}".format(episode, self.episode_length, cost, inference_time))
            inference_time = inference_time + cost

    def warmup(self):

        obs_get = self.envs.reset()
        obs = obs_get[:, 1:, 4*self.num_agents:]

        if self.use_centralized_V:
            share_obs = obs_get[:,1:,:4*self.num_agents]
        else:
            share_obs = obs[:,1:,:-self.num_agents]
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        value, action, action_log_prob, rnn_states, rnn_states_critic, hid_state, trigger_door_count\
            = self.trainer.policy.get_actions(step, np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.hid_states[step]),)

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        hid_state = np.array(np.split(_t2n(hid_state), self.n_rollout_threads))

        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':

            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            actions_env = actions

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, hid_state, trigger_door_count

    def insert(self, data):
        obs_get, rewards, dones, infos, values, actions, action_log_probs,\
            rnn_states, rnn_states_critic, hid_state = data
        obs = obs_get[:, 1:, 4*self.num_agents:]

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
                                                    dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs_get[:,1:,:4*self.num_agents]

        else:
            share_obs = obs[:, 1:, :-self.num_agents]

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks, hid_state)

    @torch.no_grad()
    def eval(self):
        for eval_step in range(10):
            self.warmup()
            eval_episode_rewards = []
            for step in range(self.episode_length):

                values, actions, action_log_probs, rnn_states, rnn_states_critic,\
                actions_env, hid_state = self.collect(step)

                obs, rewards, dones, infos = self.envs.step(actions_env)
                eval_episode_rewards.append(rewards)
                data = obs, rewards, dones, infos, values, actions, action_log_probs,\
                       rnn_states, rnn_states_critic, hid_state

                self.insert(data)
            self.buffer.after_update()
            eval_episode_rewards = np.array(eval_episode_rewards)
            eval_env_infos = {}
            eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
            eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
            print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))

    @torch.no_grad()
    def eval_kd(self):
        from mappo.utils.memory import ReplayMemory
        kd_buffer = ReplayMemory(capacity=100000, save_dir=self.save_dir, new=True)

        for eval_step in range(80):
            self.warmup()
            eval_episode_rewards = []
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic,\
                actions_env, hid_state = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                eval_episode_rewards.append(rewards)
                data = obs, rewards, dones, infos, values, actions, action_log_probs,\
                       rnn_states, rnn_states_critic, hid_state
                self.insert(data)

                hd_temp = self.buffer.hid_states[step].reshape(-1, *self.buffer.hid_states[step].shape[2:])
                hd_next_temp = self.buffer.hid_states[step+1, :, 0, :]
                hd_next_temp = hd_next_temp.reshape(-1, *hd_next_temp.shape[2:])

                obs_temp = self.buffer.obs[step].reshape(-1, *self.buffer.obs[step+1].shape[2:])
                for i in range(hd_temp.shape[0]):
                    kd_buffer.push(hd_temp[i], hd_next_temp[i],
                                   obs_temp[i])

            self.buffer.after_update()
            eval_episode_rewards = np.array(eval_episode_rewards)
            eval_env_infos = {}
            eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
            eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
            print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        kd_buffer.save_dataset()

    @torch.no_grad()
    def render(self):
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                             np.concatenate(rnn_states),
                                                             np.concatenate(masks),
                                                             deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
