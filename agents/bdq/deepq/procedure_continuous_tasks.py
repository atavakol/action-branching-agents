import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile
import datetime
import bdq.common.tf_util as U
from bdq import logger
from bdq.common.schedules import ConstantSchedule, LinearSchedule, PiecewiseSchedule
from bdq import deepq
from bdq.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import time
import pickle
import csv

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path, num_cpu=16):
        with open(path, "rb") as f:
            model_data, act_params = dill.load(f)
        act = deepq.build_act(**act_params)
        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path):
        """Save model to a pickle located at `path`"""
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data, self._act_params), f)

def load(path, num_cpu=16):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle
    num_cpu: int
        number of cpus to use for executing the policy

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path, num_cpu=num_cpu)

model_saved = False
max_eval_reward_mean = None

def learn_continuous_tasks(env,
          q_func,
          env_name,
          method_name,
          dir_path,
          time_stamp,
          total_num_episodes,
          num_actions_pad=33, 
          lr=1e-4,
          grad_norm_clipping=10,
          max_timesteps=int(1e8),
          buffer_size=int(1e6),
          exploration_fraction=None, 
          exploration_final_eps=None, 
          train_freq=1,
          batch_size=64,
          print_freq=10,
          learning_starts=1000,
          gamma=0.99,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None, 
          prioritized_replay_eps=int(1e8),
          independent=False,
          dueling=True,
          num_cpu=16,
          losses_version=2,
          epsilon_greedy=False,
          timesteps_std=1e6,
          initial_std=0.4,
          final_std=0.05,
          eval_freq=100,
          n_eval_episodes=10,
          eval_std=0.01,
          target_version="mean",
          loss_type="L2", 
          callback=None):
    """Train a branching deepq model to solve continuous control tasks via discretization.
    Current assumptions in the implementation: 
    - for solving continuous control domains via discretization (can be adjusted to be compatible with naturally disceret-action domains using 'env.action_space.n')
    - uniform number of sub-actions per action dimension (can be generalized to heterogeneous number of sub-actions across branches) 

    Parameters
    -------
    env : gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions_pad: int
        number of sub-actions per action dimension (= num of discretization grains/bars + 1)
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimize for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
        0.1 for dqn-baselines
    exploration_final_eps: float
        final value of random action probability
        0.02 for dqn-baselines 
    train_freq: int
        update the model every `train_freq` steps.
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    grad_norm_clipping: int
        set None for no clipping
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the unified TD error for updating priorities.
        Erratum: The camera-ready copy of this paper incorrectly reported 1e-8. 
        The value used to produece the results is 1e8.
    dueling: bool
        indicates whether we are using the dueling architectures or not
    num_cpu: int
        number of cpus to use for training
    losses_version: int
        optimization version number
    dir_path: str 
        path for logs and results to be stored in 
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """

    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)

    num_action_grains = num_actions_pad - 1
    num_action_dims = env.action_space.shape[0] 
    num_action_streams = num_action_dims  
    num_actions = num_actions_pad*num_action_streams # total numb network outputs for action branching with one action dimension per branch
    
    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=num_actions, 
        num_action_streams=num_action_streams,
        batch_size=batch_size,
        optimizer_name="Adam",
        learning_rate=lr,
        grad_norm_clipping=grad_norm_clipping,
        gamma=gamma,
        double_q=True,
        scope="deepq",
        reuse=None,
        losses_version=losses_version,
        independent=independent,
        dueling=dueling,
        target_version=target_version,
        loss_type="L2"
    )
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': num_actions, 
        'num_action_streams': num_action_streams, 
    }

    print('Create the log writer for TensorBoard visualizations.')
    log_dir = "{}/tensorboard_logs/{}".format(dir_path, env_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_writer = tf.summary.FileWriter('{}/{}_{}_{}'.format(log_dir, method_name, time_stamp, env_name), sess.graph)  
    score_placeholder = tf.placeholder(tf.float32, [], name='score_placeholder')
    tf.summary.scalar('score', score_placeholder)
    lr_constant = tf.constant(lr, name='lr_constant')
    tf.summary.scalar('learning_rate', lr_constant)
    version_constant = tf.constant(losses_version, name='losses_version_constant')
    tf.summary.scalar('losses_version', losses_version)
    summary_op = tf.summary.merge_all()

    eval_placeholder = tf.placeholder(tf.float32, [], name='eval_placeholder')
    eval_summary = tf.summary.scalar('evaluation', eval_placeholder)

    # Create the replay buffer 
    if prioritized_replay: 
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    if epsilon_greedy:
        approximate_num_iters = 2e6 / 4
        exploration = PiecewiseSchedule([(0, 1.0),
                                        (approximate_num_iters / 50, 0.1), 
                                        (approximate_num_iters / 5, 0.01) 
                                        ], outside_value=0.01)
    else:
        exploration = ConstantSchedule(value=0.0) # greedy policy
        std_schedule = LinearSchedule(schedule_timesteps=timesteps_std,
                                     initial_p=initial_std,
                                     final_p=final_std)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    # Initialize the parameters used for converting branching, discrete action indeces to continuous actions 
    low = env.action_space.low 
    high = env.action_space.high 
    actions_range = np.subtract(high, low) 

    episode_rewards = []
    reward_sum = 0.0
    num_episodes = 0
    time_steps = [0]
    time_spent_exploring = [0]
    
    prev_time = time.time()
    n_trainings = 0

    # Set up on-demand rendering of Gym environments using keyboard controls: 'r'ender or 's'top
    import termios, fcntl, sys
    fd = sys.stdin.fileno()
    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    render = False

    # Open a dircetory for recording results
    results_dir = "{}/results/{}".format(dir_path, env_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  

    saved_mean_reward = None
    displayed_mean_reward = None

    def evaluate(step, episode_number):
        global max_eval_reward_mean, model_saved
        print('Evaluate...')
        eval_reward_sum = 0.0
        # Run evaluation episodes
        for eval_episode in range(n_eval_episodes):
            obs = env.reset()
            done = False
            while not done:
                # Choose action
                action_idxes = np.array(act(np.array(obs)[None], stochastic=False)) # deterministic
                actions_greedy = action_idxes / num_action_grains * actions_range + low

                if eval_std == 0.0:
                    action = actions_greedy
                else:
                    action = []
                    for index in range(len(actions_greedy)): 
                        a_greedy = actions_greedy[index]
                        out_of_range_action = True 
                        while out_of_range_action:
                            a_stoch = np.random.normal(loc=a_greedy, scale=eval_std)
                            a_idx_stoch = np.rint((a_stoch + high[index]) / actions_range[index] * num_action_grains)
                            if a_idx_stoch >= 0 and a_idx_stoch < num_actions_pad:
                                action.append(a_stoch)
                                out_of_range_action = False

                # Step
                obs, rew, done, _ = env.step(action)
                eval_reward_sum += rew

        # Average the rewards and log
        eval_reward_mean = eval_reward_sum / n_eval_episodes
        print(eval_reward_mean, 'over', n_eval_episodes, 'episodes')

        with open("{}/{}_{}_{}_eval.csv".format(results_dir, method_name, time_stamp, env_name), "a") as eval_fw: 
            eval_writer = csv.writer(eval_fw, delimiter="\t", lineterminator="\n",)
            eval_writer.writerow([episode_number, step, eval_reward_mean])
        summary = sess.run(eval_summary, feed_dict={eval_placeholder: eval_reward_mean})
        log_writer.add_summary(summary, episode_number)
        log_writer.flush()

        if max_eval_reward_mean is None or eval_reward_mean > max_eval_reward_mean:
            logger.log("Saving model due to mean eval increase: {} -> {}".format(max_eval_reward_mean, eval_reward_mean))
            U.save_state(model_file)
            model_saved = True
            max_eval_reward_mean = eval_reward_mean

    with tempfile.TemporaryDirectory() as td:
        model_file = os.path.join(td, "model")

        evaluate(0, 0)
        obs = env.reset()

        with open("{}/{}_{}_{}.csv".format(results_dir, method_name, time_stamp, env_name), "w") as fw: 
            writer = csv.writer(fw, delimiter="\t", lineterminator="\n",)

            t = -1
            while True:
                t += 1
                #logger.log("Q-values: {}".format(debug["q_values"](np.array([obs])))) 
                
                # Select action and update exploration probability
                action_idxes = np.array(act(np.array(obs)[None], update_eps=exploration.value(t)))

                # Convert sub-actions indexes (discrete sub-actions) to continuous controls
                action = action_idxes / num_action_grains * actions_range + low
                
                if not epsilon_greedy: # Gaussian noise
                    actions_greedy = action
                    action_idx_stoch = []
                    action = []
                    for index in range(len(actions_greedy)): 
                        a_greedy = actions_greedy[index]
                        out_of_range_action = True 
                        while out_of_range_action:
                            # Sample from a Gaussian with mean at the greedy action and a std following a schedule of choice  
                            a_stoch = np.random.normal(loc=a_greedy, scale=std_schedule.value(t))

                            # Convert sampled cont action to an action idx
                            a_idx_stoch = np.rint((a_stoch + high[index]) / actions_range[index] * num_action_grains)

                            # Check if action is in range
                            if a_idx_stoch >= 0 and a_idx_stoch < num_actions_pad:
                                action_idx_stoch.append(a_idx_stoch)
                                action.append(a_stoch)
                                out_of_range_action = False

                    action_idxes = action_idx_stoch

                new_obs, rew, done, _ = env.step(action)

                # On-demand rendering
                if (t+1) % 100 == 0:
                    # TODO better?
                    termios.tcsetattr(fd, termios.TCSANOW, newattr)
                    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
                    try:
                        try:
                            c = sys.stdin.read(1)
                            if c == 'r':
                                print()
                                print('Rendering begins...')
                                render = True
                            elif c == 's':
                                print()
                                print('Stop rendering!')
                                render = False
                                env.render(close=True)
                        except IOError: pass
                    finally:
                        termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
                        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

                # Visualize Gym environment on render
                if render: env.render()

                # Store transition in the replay buffer
                replay_buffer.add(obs, action_idxes, rew, new_obs, float(done))
                obs = new_obs

                reward_sum += rew
                if done:
                    obs = env.reset()
                    time_spent_exploring[-1] = int(100 * exploration.value(t))
                    time_spent_exploring.append(0)
                    episode_rewards.append(reward_sum)
                    time_steps[-1] = t
                    reward_sum = 0.0
                    time_steps.append(0)
                    # Frequently log to file
                    writer.writerow([len(episode_rewards), t, episode_rewards[-1]])

                if t > learning_starts and t % train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer
                    if prioritized_replay:
                        experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights) #np.ones_like(rewards)) #TEMP AT NEW
                    if prioritized_replay:
                        new_priorities = np.abs(td_errors) + prioritized_replay_eps
                        replay_buffer.update_priorities(batch_idxes, new_priorities)

                    n_trainings += 1

                if t > learning_starts and t % target_network_update_freq == 0:
                    # Update target network periodically
                    update_target()

                if len(episode_rewards) == 0:    mean_100ep_reward = 0
                elif len(episode_rewards) < 100: mean_100ep_reward = np.mean(episode_rewards)
                else:                            mean_100ep_reward = np.mean(episode_rewards[-100:])

                num_episodes = len(episode_rewards)
                if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    current_time = time.time()
                    logger.record_tabular("trainings per second", n_trainings / (current_time - prev_time))
                    logger.dump_tabular()
                    n_trainings = 0
                    prev_time = current_time

                    # Write recent log for TensorBoard
                    for prev_i in reversed(range(print_freq)):
                        summary = sess.run(summary_op, feed_dict={score_placeholder: episode_rewards[-prev_i-1]})
                        log_writer.add_summary(summary, num_episodes-prev_i)
                    log_writer.flush()
                    
                if t > learning_starts and num_episodes > 100:
                    if displayed_mean_reward is None or mean_100ep_reward > displayed_mean_reward:
                        if print_freq is not None:
                            logger.log("Mean reward increase: {} -> {}".format(
                                displayed_mean_reward, mean_100ep_reward))
                        displayed_mean_reward = mean_100ep_reward    

                # Performance evaluation with a greedy policy
                if done and num_episodes % eval_freq == 0:
                    evaluate(t+1, num_episodes)
                    obs = env.reset()

                # STOP training
                if num_episodes >= total_num_episodes:
                    break

            if model_saved:
                logger.log("Restore model with mean eval: {}".format(max_eval_reward_mean))
                U.load_state(model_file)

    data_to_log = {
       'time_steps' : time_steps,
       'episode_rewards' : episode_rewards,
       'time_spent_exploring' : time_spent_exploring
    }
    
    # Write to file the episodic rewards, number of steps, and the time spent exploring
    with open("{}/{}_{}_{}.txt".format(results_dir, method_name, time_stamp, env_name), 'wb') as fp:
        pickle.dump(data_to_log, fp)

    return ActWrapper(act, act_params)