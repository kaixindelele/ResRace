import numpy as np
import tensorflow.compat.v1 as tf
import gym
import time
import math
import pybullet
import matplotlib.pyplot as plt
import spinup.algos.tf1.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from utils.functions import modify_obs
from racecar_gym import SingleAgentScenario
from racecar_gym.envs import SingleAgentRaceEnv
from utils.config import Config, APF_Config
from mapf import Nonlinear_Controller

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

num_workers = 1
manual_action_dim = Config['action_dim']
imu_dim = Config['imu_dim']
manual_obs_dim = Config['lidar_dim'] + imu_dim + Config['action_dim']
show_flag, new_train_flag, save_flag = True, True, True
display_frequency = 1.1
render_pause = 0
track = 'plechaty'
train_mode = 'resrace_ppo'
wait_frame = 180
exp_prove = 0.1
still_threshold = 1/45
save_path = './logs/' + train_mode + '/' + track
EPS = 1e-8
lap_reward = 0


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]

def learning_curve_display(epoch, last_show_num, logger, train_pro_list):
    train_pro_list.append(np.mean(logger.epoch_dict['EpPro']))
    if epoch / last_show_num > display_frequency:
        plt.cla()
        plt.title(track + train_mode, loc='center')
        plt.plot(train_pro_list, label="train_progress")
        plt.legend()
        plt.pause(0.1)
        last_show_num = epoch
    return train_pro_list, last_show_num


def ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=10000, epochs=50, gamma=1, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=5000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=1):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = manual_obs_dim
    act_dim = manual_action_dim
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholder(manual_obs_dim), core.placeholder(manual_action_dim)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    #-----------------------------------------------------------------#
    saver = tf.train.Saver()
    network_file_name = save_path + ".ckpt"
    last_show_num = 1
    if new_train_flag:
        print("!!!!!!-------- Attention: Begin a NEW TRAINING --------!!!!!!")
        train_pro_list, test_pro_list = [], []
    else:
        print("!!!!!!------ Attention: Inherit PERVIOUS TRAINING ------!!!!!!")
        saver.restore(sess, network_file_name)
        train_pro_list = np.load(save_path+'train.npy').tolist()
        start_steps = 0
        network = tf.trainable_variables()
        variable_name = [v.name for v in tf.trainable_variables()]

    o_raw, ep_ret, ep_len = env.reset(), 0, 0

    last_loc, last_steering, last_motor = o_raw['pose'][:3], 0, 0
    lidar_obs = o_raw['lidar']
    o = modify_obs(o_raw)

    progress_history, velo_history = [0]*wait_frame, [0]*wait_frame
    total_steps = steps_per_epoch * epochs
    last_total_time = 0
    epoch, agent_lap, last_progress, lap_counter = 0, 0, 0, 0

    eps_pro_list, epo_pro_list = [], []
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})
            a = a[0]
            #----------------------------------------------------------------------------------------------------------#
            a_residual = a
            a_mapf = Nonlinear_Controller(lidar_obs, rep_range=APF_Config[track])
            a = np.clip((a_mapf+a_residual), -1, +1)
            o2_raw, r, d, _ = env.step(a)
            d = False
            current_loc = o2_raw['pose'][:3]
            delta_loc, delta_steering, delta_motor = math.sqrt(np.sum(np.square(current_loc-last_loc))), abs(a[0]-last_steering), abs(a[1]-last_motor)
            lidar_obs = o2_raw['lidar']
            velocity_bonus = 0.5 * math.sqrt(np.sum(np.square(o2_raw['velocity'][:2])))
            steering_penalty = -0.2 * (delta_steering+abs(a[0]))
            motor_penalty = -0.5 * (delta_motor+1-a[1])
            o2 = modify_obs(o2_raw, a)
            # r += (steering_penalty+motor_penalty)
            # r += velocity_bonus
            last_loc, last_steering, last_motor = current_loc, a[0], a[1]

            agent_state = env.scenario.world.state()
            agent_progress = agent_state['A']['progress']

            delta_progress = agent_progress - last_progress
            velo_history = velo_history[1:] + [delta_loc]
            if (ep_len > wait_frame and sum(velo_history)/wait_frame < still_threshold):
                d = True
                print('Agent Died: For Collision or Long-time Brake.')
            if agent_progress <=0.05 and last_progress>=0.95 and lap_counter >= 450:
                agent_lap += 1
                r += lap_reward
                lap_counter = 0
                print('Training Agent: Finish one LAP.')
            elif track != 'montreal':
                progress_history = progress_history[1:] + [delta_progress]
                if (ep_len > wait_frame and sum(progress_history) < -0.1):
                    d = True
                    print('Agent Died: For Wrong Way.')
            
            last_progress = agent_progress

            ep_ret += r
            ep_len += 1
            lap_counter += 1
            #----------------------------------------------------------------------------------------------------------#
            # if not d and ep_len == max_ep_len:
            # r += agent_progress*lap_reward
            # save and log
            buf.store(o, a_residual, r, v_t, logp_t)
            logger.store(VVals=v_t)
            # Update obs (critical!)
            o = o2

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                logger.store(EpPro=(agent_progress+agent_lap))
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if d else sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0
                o = modify_obs(o_raw)
                agent_progress, agent_lap, last_progress, lap_counter = 0, 0, 0, 0

        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and not save_flag:
            log_path = saver.save(sess, network_file_name)
            np.save(save_path+'train.npy', train_pro_list)
            print("Save to path: ", log_path)

        # Perform PPO update!
        update()
        train_pro_list,  last_show_num = learning_curve_display(epoch, last_show_num, logger, train_pro_list)
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpPro', average_only=True)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    scenario = SingleAgentScenario.from_spec("./scenarios/"+track+".yml", rendering=show_flag)
    env = SingleAgentRaceEnv(scenario=scenario)
    env.action_space = gym.spaces.Box(-1, 1, shape=(manual_action_dim,))
    env.reset()
    pybullet.setGravity(0, 0, -9.8)
    ppo(lambda : env, actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
