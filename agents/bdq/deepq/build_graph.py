"""Deep Q-learning graph

The functions in this script are used to create the following functionalities:

======= act ========

    Function to choose an action given an observation

    Parameters
    ----------
    observation: object
        observation that can be fed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative no update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)


======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q-learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    where Q' is lagging behind Q to stablize the learning. For example for Atari
    Q' is set to Q once every 10000 updates training steps.
"""

import tensorflow as tf
import bdq.common.tf_util as U
import numpy as np


def build_act(make_obs_ph, q_func, num_actions, num_action_streams, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
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
    num_actions: int
        total number of sub-actions to be represented at the output 
    num_action_streams: int
        specifies the number of action branches in action value (or advantage) function representation
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select an action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic") 
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        
        assert (num_action_streams >= 1), "number of action branches is not acceptable, has to be >=1"
        
        # TODO better: enable non-uniform number of sub-actions per joint
        num_actions_pad = num_actions//num_action_streams # number of sub-actions per action dimension
        
        output_actions = []
        for dim in range(num_action_streams):
            q_values_batch = q_values[dim][0] # TODO better: does not allow evaluating actions over a whole batch
            deterministic_action = tf.argmax(q_values_batch)
            random_action = tf.random_uniform([], minval=0, maxval=num_actions//num_action_streams, dtype=tf.int64)
            chose_random = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32) < eps
            stochastic_action = tf.cond(chose_random, lambda: random_action, lambda: deterministic_action)
            output_action = tf.cond(stochastic_ph, lambda: stochastic_action, lambda: deterministic_action)
            output_actions.append(output_action)
        
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps)) 

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        return act


def build_train(make_obs_ph, q_func, num_actions, num_action_streams, batch_size, optimizer_name, learning_rate, grad_norm_clipping=None, gamma=0.99, double_q=True, scope="deepq", reuse=None, losses_version=2, independent=False, dueling=True, target_version="mean", loss_type="L2"):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
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
    num_actions: int
        total number of sub-actions to be represented at the output  
    num_action_streams: int
        specifies the number of action branches in action value (or advantage) function representation
    batch_size: int
        size of the sampled mini-batch from the replay buffer 
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for deep Q-learning 
    grad_norm_clipping: float or None
        clip graident norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q-Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled. BDQ uses it. 
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    losses_version: int
        specifies the version number for merging of losses across the branches
        version 2 is the best-performing loss used for BDQ.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select an action given an observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """

    assert independent and losses_version == 4 or not independent, 'independent needs to be used along with loss v4'
    assert independent and target_version == "indep" or not independent, 'independent needs to be used along with independent TD targets'

    act_f = build_act(make_obs_ph, q_func, num_actions, num_action_streams, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # Set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None, num_action_streams], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # Q-network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True) # reuse parameters from act
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # Target Q-network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        if double_q: 
            selection_q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
        else: 
            selection_q_tp1 = q_tp1

        num_actions_pad = num_actions//num_action_streams

        q_values = []
        for dim in range(num_action_streams):
            selected_a = tf.squeeze(tf.slice(act_t_ph, [0, dim], [batch_size, 1])) # TODO better?
            q_values.append(tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * q_t[dim], axis=1))
        
        if target_version == "indep":
            target_q_values = []
            for dim in range(num_action_streams):
                selected_a = tf.argmax(selection_q_tp1[dim], axis=1)
                selected_q = tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * q_tp1[dim], axis=1)
                masked_selected_q = (1.0 - done_mask_ph) * selected_q
                target_q = rew_t_ph + gamma * masked_selected_q
                target_q_values.append(target_q)
        elif target_version == "max":
            for dim in range(num_action_streams):
                selected_a = tf.argmax(selection_q_tp1[dim], axis=1)
                selected_q = tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * q_tp1[dim], axis=1) 
                masked_selected_q = (1.0 - done_mask_ph) * selected_q
                if dim == 0:
                    max_next_q_values = masked_selected_q
                else:
                    max_next_q_values = tf.maximum(max_next_q_values, masked_selected_q)
            target_q_values = [rew_t_ph + gamma * max_next_q_values] * num_action_streams # TODO better?
        elif target_version == "mean":
            for dim in range(num_action_streams):
                selected_a = tf.argmax(selection_q_tp1[dim], axis=1)
                selected_q = tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * q_tp1[dim], axis=1) 
                masked_selected_q = (1.0 - done_mask_ph) * selected_q
                if dim == 0:
                    mean_next_q_values = masked_selected_q
                else:
                    mean_next_q_values += masked_selected_q 
            mean_next_q_values /= num_action_streams
            target_q_values = [rew_t_ph + gamma * mean_next_q_values] * num_action_streams # TODO better?
        else:
            assert False, 'unsupported target version ' + str(target_version)

        if optimizer_name == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            assert False, 'unsupported optimizer ' + str(optimizer_name)

        if loss_type == "L2":
            loss_function = tf.square
        elif loss_type == "Huber":
            loss_function = U.huber_loss
        else:
            assert False, 'unsupported loss type ' + str(loss_type)

        if losses_version == 1:
            mean_q_value = sum(q_values) / num_action_streams
            mean_target_q_value = sum(target_q_values) / num_action_streams
            td_error = mean_q_value - tf.stop_gradient(mean_target_q_value)
            loss = loss_function(td_error)
            weighted_mean_loss = tf.reduce_mean(importance_weights_ph * loss)
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_mean_loss,
                                                var_list=q_func_vars,
                                                total_n_streams=(num_action_streams + (1 if dueling else 0)),
                                                clip_val=grad_norm_clipping)
            optimize_expr = [optimize_expr]

        elif losses_version == 5:
            for dim in range(num_action_streams):
                abs_dim_td_error = tf.abs(q_values[dim] - tf.stop_gradient(target_q_values[dim]))
                if dim == 0:
                    td_error = abs_dim_td_error
                else:
                    td_error += abs_dim_td_error
            td_error /= num_action_streams
            loss = loss_function(td_error)
            weighted_mean_loss = tf.reduce_mean(importance_weights_ph * loss)
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_mean_loss,
                                                var_list=q_func_vars,
                                                total_n_streams=(num_action_streams + (1 if dueling else 0)),
                                                clip_val=grad_norm_clipping)
            optimize_expr = [optimize_expr]

        elif losses_version in [2, 3, 4]:
            stream_losses = []
            for dim in range(num_action_streams):
                dim_td_error = q_values[dim] - tf.stop_gradient(target_q_values[dim])
                dim_loss = loss_function(dim_td_error)
                # Scaling of learning based on importance sampling weights is optional, either way works 
                stream_losses.append(tf.reduce_mean(dim_loss * importance_weights_ph)) # with scaling
                #stream_losses.append(tf.reduce_mean(dim_loss)) # without scaling 
                if dim == 0:
                    td_error = tf.abs(dim_td_error)  
                else:
                    td_error += tf.abs(dim_td_error) 
            #td_error /= num_action_streams 

            if losses_version == 2:
                mean_loss = sum(stream_losses) / num_action_streams
                optimize_expr = U.minimize_and_clip(optimizer,
                                                    mean_loss,
                                                    var_list=q_func_vars,
                                                    total_n_streams=(num_action_streams + (1 if dueling else 0)),
                                                    clip_val=grad_norm_clipping)
                optimize_expr = [optimize_expr]
            elif losses_version == 3:
                optimize_expr = []
                for dim in range(num_action_streams):
                    optimize_expr.append(U.minimize_and_clip(optimizer,
                                                             stream_losses[dim],
                                                             var_list=q_func_vars,
                                                             total_n_streams=(num_action_streams + (1 if dueling else 0)),
                                                             clip_val=grad_norm_clipping))
            else: # losses_version = 4
                optimize_expr = []
                for dim in range(num_action_streams):
                    if optimizer_name == "Adam":
                        optimizer = tf.train.AdamOptimizer(learning_rate)
                    else:
                        assert False, 'optimizer type not supported'
                    optimize_expr.append(U.minimize_and_clip(optimizer, 
                                                             stream_losses[dim],
                                                             var_list=q_func_vars,
                                                             total_n_streams=(1 if independent else num_action_streams + (1 if dueling else 0)),
                                                             clip_val=grad_norm_clipping))
        else:
            assert False, 'unsupported loss version ' + str(losses_version)

        # Target Q-network parameters are periodically updated with the Q-network's
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=optimize_expr
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values}