import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def _mlp_branching(hiddens_common, hiddens_actions, hiddens_value, independent, num_action_branches, dueling, aggregator, distributed_single_stream, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt

        if dueling:
            assert (aggregator in ['reduceLocalMean','reduceGlobalMean','naive','reduceLocalMax','reduceGlobalMax']), 'appropriate aggregator method needs be set when using dueling architecture'
            assert (hiddens_value), 'state-value network layer size cannot be empty when using dueling architecture'
        else: 
            assert (aggregator is None), 'no aggregator method to be set when not using dueling architecture'
            assert (not hiddens_value), 'state-value network layer size has to be empty when not using dueling architecture'

        if num_action_branches < 2 and independent: 
            assert False, 'independent only makes sense when there are more than one action dimension'

        # Create the shared network module (unless independent)
        with tf.variable_scope('common_net'):
            if not independent: 
                for hidden in hiddens_common:
                    out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
            else: 
                if hiddens_common != []:
                    total_indep_common_out = []
                    for action_stream in range(num_action_branches):
                        indep_common_out = out
                        for hidden in hiddens_common:
                            indep_common_out = layers.fully_connected(indep_common_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                        total_indep_common_out.append(indep_common_out)
                    out = total_indep_common_out
                else:
                    out = [out] * num_action_branches
                
        # Create the action branches
        with tf.variable_scope('action_value'):
            if not independent:
                if (not distributed_single_stream or num_action_branches == 1):
                    total_action_scores = []
                    for action_stream in range(num_action_branches):
                        action_out = out
                        for hidden in hiddens_actions:
                            action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)    
                        action_scores = layers.fully_connected(action_out, num_outputs=num_actions//num_action_branches, activation_fn=None)
                        if aggregator == 'reduceLocalMean':
                            assert dueling, 'aggregation only needed for dueling architectures'
                            action_scores_mean = tf.reduce_mean(action_scores, 1)
                            total_action_scores.append(action_scores - tf.expand_dims(action_scores_mean, 1))
                        elif aggregator == 'reduceLocalMax':
                            assert dueling, 'aggregation only needed for dueling architectures'
                            action_scores_max = tf.reduce_max(action_scores, 1)
                            total_action_scores.append(action_scores - tf.expand_dims(action_scores_max, 1))
                        else:
                            total_action_scores.append(action_scores)
                elif distributed_single_stream: # TODO better: implementation of single-stream case
                    action_out = out
                    for hidden in hiddens_actions:
                        action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)    
                    action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)
                    if aggregator == 'reduceLocalMean':
                        assert dueling, 'aggregation only needed for dueling architectures' 
                        total_action_scores = []
                        for action_stream in range(num_action_branches):
                            # Slice action values (or advantages) of each action dimension and locally subtract their mean
                            sliced_actions_of_dim = tf.slice(action_scores, [0,action_stream*num_actions//num_action_branches], [-1,num_actions//num_action_branches])
                            sliced_actions_mean = tf.reduce_mean(sliced_actions_of_dim, 1)
                            sliced_actions_centered = sliced_actions_of_dim - tf.expand_dims(sliced_actions_mean, 1)
                            total_action_scores.append(sliced_actions_centered)
                    elif aggregator == 'reduceLocalMax':
                        assert dueling, 'aggregation only needed for dueling architectures'
                        total_action_scores = []
                        for action_stream in range(num_action_branches):
                            # Slice action values (or advantages) of each action dimension and locally subtract their max
                            sliced_actions_of_dim = tf.slice(action_scores, [0,action_stream*num_actions//num_action_branches], [-1,num_actions//num_action_branches])
                            sliced_actions_max = tf.reduce_max(sliced_actions_of_dim, 1)
                            sliced_actions_centered = sliced_actions_of_dim - tf.expand_dims(sliced_actions_max, 1)
                            total_action_scores.append(sliced_actions_centered)
                    else:             
                        total_action_scores = action_scores
            else:
                if (not distributed_single_stream or num_action_branches == 1):
                    total_action_scores = []
                    for action_stream in range(num_action_branches):
                        action_out = total_indep_common_out[action_stream] 
                        for hidden in hiddens_actions:
                            action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)    
                        action_scores = layers.fully_connected(action_out, num_outputs=num_actions//num_action_branches, activation_fn=None)
                        if aggregator == 'reduceLocalMean':
                            assert dueling, 'aggregation only needed for dueling architectures'
                            action_scores_mean = tf.reduce_mean(action_scores, 1)
                            total_action_scores.append(action_scores - tf.expand_dims(action_scores_mean, 1))
                        elif aggregator == 'reduceLocalMax':
                            assert dueling, 'aggregation only needed for dueling architectures'
                            action_scores_max = tf.reduce_max(action_scores, 1)
                            total_action_scores.append(action_scores - tf.expand_dims(action_scores_max, 1))
                        else:
                            total_action_scores.append(action_scores)
                elif distributed_single_stream: # TODO better: implementation of single-stream case    
                    pass 
        
        if dueling: # create a separate state-value branch
            if not independent: 
                with tf.variable_scope('state_value'):
                    state_out = out
                    for hidden in hiddens_value:
                        state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                    state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                if aggregator == 'reduceLocalMean': 
                    # Local centering wrt branch's mean value has already been done
                    action_scores_adjusted = total_action_scores
                elif aggregator == 'reduceGlobalMean': 
                    action_scores_mean = sum(total_action_scores) / num_action_branches
                    action_scores_adjusted = total_action_scores - tf.expand_dims(action_scores_mean, 1)
                elif aggregator == 'reduceLocalMax':
                    # Local max-reduction has already been done       
                    action_scores_adjusted = total_action_scores        
                elif aggregator == 'reduceGlobalMax':
                    assert False, 'not implemented'
                    action_scores_max = max(total_action_scores)
                    action_scores_adjusted = total_action_scores - tf.expand_dims(action_scores_max, 1)
                elif aggregator == 'naive':
                    action_scores_adjusted = total_action_scores 
                else:
                    assert (aggregator in ['reduceLocalMean','reduceGlobalMean','naive','reduceLocalMax','reduceGlobalMax']), 'aggregator method is not supported' 
                return [state_score + action_score_adjusted for action_score_adjusted in action_scores_adjusted]

            elif independent and num_action_branches > 1:               
                with tf.variable_scope('state_value'):
                    total_state_scores = []
                    for action_stream in range(num_action_branches):
                        state_out = total_indep_common_out[action_stream] 
                        for hidden in hiddens_value:
                            state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                        state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                        total_state_scores.append(state_score)
                    if aggregator == 'reduceLocalMean': 
                        action_scores_adjusted = total_action_scores # local centering wrt branch's mean value has already been done
                    elif aggregator == 'reduceGlobalMean': 
                        action_scores_mean = sum(total_action_scores) / num_action_branches
                        action_scores_adjusted = total_action_scores - tf.expand_dims(action_scores_mean, 1)
                    elif aggregator == 'reduceLocalMax':        
                        action_scores_adjusted = total_action_scores # local max-reduction has already been done        
                    elif aggregator == 'reduceGlobalMax':
                        assert False, 'Not implemented!'
                        action_scores_max = max(total_action_scores)
                        action_scores_adjusted = total_action_scores - tf.expand_dims(action_scores_max, 1)
                    elif aggregator == 'naive':
                        action_scores_adjusted = total_action_scores  

                    q_values_out = []
                    num_actions_pad = num_actions//num_action_branches
                    for action_stream in range(num_action_branches):
                        for a_score in action_scores_adjusted[action_stream*num_actions_pad:num_actions_pad*(num_action_branches+1)]: 
                            q_values_out.append(total_state_scores[action_stream] + a_score)
                return q_values_out
        else:
            return total_action_scores
   
def mlp_branching(hiddens_common=[], hiddens_actions=[], hiddens_value=[], independent=False, num_action_branches=None, dueling=True, aggregator='reduceLocalMean', distributed_single_stream=False):
    """This model takes as input an observation and returns values of all sub-actions -- either by 
    combining the state value and the sub-action advantages (i.e. dueling), or directly the Q-values.
    
    Parameters
    ----------
    hiddens_common: [int]
        list of sizes of hidden layers in the shared network module -- 
        if this is an empty list, then the learners across the branches 
        are considered 'independent'

    hiddens_actions: [int]
        list of sizes of hidden layers in the action-value/advantage branches -- 
        currently assumed the same across all such branches 

    hiddens_value: [int]
        list of sizes of hidden layers for the state-value branch 

    num_action_branches: int
        number of action branches (= num_action_dims in current implementation)

    dueling: bool
        if using dueling, then the network structure becomes similar to that of 
        dueling (i.e. Q = f(V,A)), but with N advantage branches as opposed to only one, 
        and if not dueling, then there will be N branches of Q-values  

    aggregator: str
        aggregator method used for dueling architecture: {naive, reduceLocalMean, reduceLocalMax, reduceGlobalMean, reduceGlobalMax}

    distributed_single_stream: bool
        True if action value (or advantage) function representation is branched (vs. combinatorial), but 
        all sub-actions are represented on the same fully-connected stream 

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp_branching(hiddens_common, hiddens_actions, hiddens_value, independent, num_action_branches, dueling, aggregator, distributed_single_stream, *args, **kwargs)


def _mlp(hiddens, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def mlp(hiddens=[]):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, *args, **kwargs)


def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            return state_score + action_scores_centered
        else:
            return action_scores
        return out

def cnn_to_mlp(convs, hiddens, dueling=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, *args, **kwargs)


def _mlp_duel(hiddens_common, hiddens, dueling, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("common_net"):
            for hidden in hiddens_common:
                out = layers.fully_connected(out,
                                             num_outputs=hidden,
                                             activation_fn=tf.nn.relu)

        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            return state_score + action_scores_centered
        else:
            return action_scores
        return out

def mlp_duel(hiddens_common=[],hiddens=[], dueling=True):
    """This model takes as input an observation and returns values of all actions
    by combining value of state and advantages of actions at that state. 

    Parameters
    ----------
    hiddens_common: [int]
        list of sizes of hidden layers part of the common net among the two streams

    hiddens: [int]
        list of sizes of hidden layers for the streams (at the moment they'll be the same)

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp_duel(hiddens_common, hiddens, dueling, *args, **kwargs)