# -*- coding: utf-8 -*-
import tensorflow as tf
import math

from derivative_books import random_black_scholes_put_call_book
from environments import DerivativeBookHedgeEnv
from networks import SequentialNeuralNetwork
from metrics import CumulativeRewardMetric
from train import ReplayBuffer
from timestep import ActionStep
from policies import BlackScholesDeltaPolicy

# ==============================================================================
# === hyperparameters
batch_size = 100
num_hedges = 12
max_episodes = 1000

book_batch_size = 1000
cost_scale = 0.

actor_num_layers = 2
actor_num_units = 10

critic_num_layers = 2
critic_num_units = 10

replay_buffer_maxlen = 1000
discount = 1.

alpha = 0.001
eps_max = 0.5
eps_min = 0.005

test_size = 100000

# ==============================================================================
# === environment
book_size, market_size, num_brownian_motions = 1, 1, 1
init_state, book = random_black_scholes_put_call_book(
    book_batch_size, market_size, num_brownian_motions)
env = DerivativeBookHedgeEnv(
    book, init_state, num_hedges, cost_scale, book_batch_size)
metrics = [CumulativeRewardMetric()]

# ==============================================================================
# === define actor and critic
actor = SequentialNeuralNetwork(
            env.observation_dimension,
            actor_num_layers,
            actor_num_units,
            env.action_dimension,
            name="actor")

critic = SequentialNeuralNetwork(
    env.observation_dimension + env.action_dimension,
    critic_num_layers,
    critic_num_units,
    1,
    name="critic")


# ==============================================================================
# === TRAAAAAAIIIIIINNN
optimizer = tf.keras.optimizers.Adam()
replay_buffer = ReplayBuffer(replay_buffer_maxlen)

# === init replay buffer
episode = 0
while episode < replay_buffer_maxlen:
    time_step = env.reset()

    while not time_step.terminated:
        action_step = ActionStep(actor(time_step.observation))
        noise_dimension = (env.batch_size, env.action_dimension)
        noise = tf.random.normal(noise_dimension, 0., eps_max)
        action_step.action += noise
        next_time_step = env.step(action_step)

        replay_buffer.add(time_step, action_step, next_time_step)

        time_step = next_time_step
    episode += 1

# ===
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

episode = 0
decay = 0
while episode < max_episodes:
    time_step = env.reset()
    cumulative_reward = 0

    while not time_step.terminated:
        action_step = ActionStep(actor(time_step.observation))

        decay += 1
        epsilon = eps_min + (eps_max - eps_min) * math.exp(-alpha * decay)
        noise_dimension = (env.batch_size, env.action_dimension)
        noise = tf.random.normal(noise_dimension, 0., epsilon)
        action_step.action += noise
        next_time_step = env.step(action_step)

        # load information in metrics
        for metric in metrics:
            metric.load(time_step, action_step)
        cumulative_reward += tf.reduce_mean(time_step.reward).numpy()

        # store in replay buffer
        replay_buffer.add(time_step, action_step, next_time_step)

        # sample mini-batch
        observation, action, reward, next_observation \
            = replay_buffer.sample(batch_size)
        next_action = actor(next_observation, True) # TODO replace by target network. Remove True?

        critit_input = tf.concat([next_observation, next_action], 1)
        critic_output = critic(critit_input, True) # TODO replace by target network. Remove True?
        critic_target = reward[:, tf.newaxis] + discount * critic_output

        # update weights
        critic_input = tf.concat([observation, action], 1)
        with tf.GradientTape(persistent=True) as tape:
            critic_output = critic(critic_input, True)
            critic_loss = tf.reduce_mean(tf.square(critic_target - critic_output))

            actor_output = actor(observation, True)
            tape.watch(actor_output)
            critic_input = tf.concat([observation, actor_output], 1)
            critic_output = critic(critic_input, True)

        critic_loss_grad = tape.gradient(critic_loss, critic.trainable_weights)
        actor_grad = tape.jacobian(actor_output, actor.trainable_weights)
        critic_grad = tape.gradient(critic_output, actor_output)
        actor_loss_grad = []
        del tape

        for gradient in actor_grad:
            expanded = critic_grad[..., tf.newaxis]
            while expanded.ndim < gradient.ndim:
                expanded = expanded[..., tf.newaxis]
            actor_loss_grad.append(
                -tf.reduce_sum(expanded * gradient, (0, 1)) / batch_size) # TODO sign change correct?

        optimizer.apply_gradients(zip(critic_loss_grad, critic.trainable_weights))
        optimizer.apply_gradients(zip(actor_loss_grad, actor.trainable_weights))

        time_step = next_time_step

    print(f"episode {episode + 1}".ljust(15) \
          + f"average reward: {cumulative_reward: .3f}")
    episode += 1

# =============================================================================
# === Test
test_env = DerivativeBookHedgeEnv(
    book, init_state, num_hedges, cost_scale, test_size)
benchmark = BlackScholesDeltaPolicy(book)
test_metric =[]

time_step = test_env.reset()
while not time_step.terminated:
    action_step = ActionStep(actor(time_step.observation))
    next_time_step = test_env.step(action_step)
    time_step = next_time_step

    test_metric.append(time_step.reward[:, tf.newaxis])

test_metric_benchmark =[]
time_step = test_env.reset()
while not time_step.terminated:
    action_step = benchmark.action(time_step)
    next_time_step = test_env.step(action_step)
    time_step = next_time_step

    test_metric_benchmark.append(time_step.reward[:, tf.newaxis])

agent_data = tf.concat(test_metric, 1)
agent_mean = tf.reduce_mean(agent_data, 1)
agent_std = tf.math.reduce_std(agent_data, 1)

benchmark_data = tf.concat(test_metric_benchmark, 1)
benchmark_mean = tf.reduce_mean(benchmark_data, 1)
benchmark_std = tf.math.reduce_std(benchmark_data, 1)