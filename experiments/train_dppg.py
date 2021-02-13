# -*- coding: utf-8 -*-
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from collections import deque

from tensorflow.keras.activations import linear, tanh

from derivative_books import random_black_scholes_put_call_book
from environments import DerivativeBookHedgeEnv
from networks import SequentialNeuralNetwork
from metrics import TrainMetric, TestMetric
from train import ReplayBuffer
from timestep import ActionStep
from policies import BlackScholesDeltaPolicy

# ==============================================================================
# === hyperparameters
batch_size = 256
max_episodes = 1000
num_hedges_a_year = 52

book_batch_size = 1
cost_scale = 0.

actor_num_layers = 3
actor_num_units = 20

critic_num_layers = 3
critic_num_units = 20

replay_buffer_maxlen = 100000
discount = 1.

alpha = 0.
eps_max = 0.05
eps_min = 0.005

test_size = 100000

actor_learning_rate = 0.0001
critic_learning_rate = 0.001
tau = 0.001

# ==============================================================================
# === visualisation
running_avg_length = 30

# ==============================================================================
# === environment
book_size, market_size, num_brownian_motions = 1, 1, 1
init_state, book = random_black_scholes_put_call_book(
    book_size, market_size, num_brownian_motions, 420)
num_hedges = math.ceil(num_hedges_a_year * book.maturity)

env = DerivativeBookHedgeEnv(
    book, init_state, num_hedges, cost_scale, book_batch_size)
metrics = [TrainMetric(max_episodes)]

benchmark = BlackScholesDeltaPolicy(book)

# ==============================================================================
# === define actor and critic
actor = SequentialNeuralNetwork(
            env.observation_dimension,
            actor_num_layers,
            actor_num_units,
            env.action_dimension,
            tanh,
            name="actor")

critic = SequentialNeuralNetwork(
    env.observation_dimension + env.action_dimension,
    critic_num_layers,
    critic_num_units,
    1,
    linear,
    name="critic")

actor_target = SequentialNeuralNetwork(
            env.observation_dimension,
            actor_num_layers,
            actor_num_units,
            env.action_dimension,
            tanh,
            name="actor")

critic_target = SequentialNeuralNetwork(
    env.observation_dimension + env.action_dimension,
    critic_num_layers,
    critic_num_units,
    1,
    linear,
    name="critic")

for model in [actor, critic, actor_target, critic_target]:
    model.build((None, model.input_dim))

actor_target.set_weights(actor.get_weights())
critic_target.set_weights(critic.get_weights())

# =============================================================================
# ===
@tf.function
def get_gradients(actor, critic, observation, action, target):
    critic_input = tf.concat([observation, action], 1)
    with tf.GradientTape(persistent=True) as tape:
        critic_output = critic(critic_input, True)
        critic_loss = tf.reduce_mean(tf.square(target - critic_output))

        actor_output = actor(observation, True)
        tape.watch(actor_output)
        critic_input = tf.concat([observation, actor_output], 1)
        critic_output = critic(critic_input, True)

    critic_loss_grad = tape.gradient(critic_loss, critic.trainable_weights)
    critic_grad = tape.gradient(critic_output, actor_output)
    actor_loss_grad = tf.gradients(actor_output, actor.trainable_weights, -critic_grad)

    batch_size = observation.shape[0]
    actor_loss_grad = list(map(lambda x: tf.divide(x, batch_size), actor_loss_grad))

    # actor_grad = tape.jacobian(actor_output, actor.trainable_weights)
    # del tape

    # batch_size = observation.shape[0]
    # actor_loss_grad = []
    # for gradient in actor_grad:
    #     expanded = critic_grad[..., tf.newaxis]
    #     while len(expanded.shape) < len(gradient.shape):
    #         expanded = expanded[..., tf.newaxis]
    #     actor_loss_grad.append(
    #         -tf.reduce_sum(expanded * gradient, (0, 1)) / batch_size) # TODO sign change correct?

    return actor_loss_grad, critic_loss_grad


def update_target_weights(actor, critic, actor_target, critic_target):
    for model, model_target in [[actor, actor_target], [critic, critic_target]]:
        weight = []
        new_weights = model.get_weights()
        old_weights = model_target.get_weights()
        for nw, ow in zip(new_weights, old_weights):
            weight.append(tau * nw + (1 - tau) * ow)

        model_target.set_weights(weight)


# ==============================================================================
# === Initialise replay buffer
replay_buffer = ReplayBuffer(replay_buffer_maxlen)
buffer_env = DerivativeBookHedgeEnv(
    book, init_state, num_hedges, cost_scale, replay_buffer_maxlen)

time_step = buffer_env.reset()
random_actions = tf.random.normal((buffer_env.batch_size, env.action_dimension, env.num_hedges))
idx = 0

while not time_step.terminated:
    action_step = ActionStep(random_actions[..., idx])
    next_time_step = buffer_env.step(action_step)
    replay_buffer.add(time_step, action_step, next_time_step)
    time_step = next_time_step
    idx += 1


# ==============================================================================
# === TRAAAAIIIIIIN
actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)

episode = 0
decay = 0
running_reward = deque(maxlen=running_avg_length)
running_error = deque(maxlen=running_avg_length)

while episode < max_episodes:
    time_step = env.reset()
    cumulative_reward = 0
    on_policy_error = 0

    while not time_step.terminated:
        action_step = ActionStep(actor(time_step.observation, True))
        on_policy_error += tf.reduce_sum(tf.square(action_step.action - benchmark.action(time_step).action)).numpy()

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
        idx_split = [env.observation_dimension, env.action_dimension, 1, env.observation_dimension]
        observation, action, reward, next_observation \
            = tf.split(replay_buffer.sample(batch_size), idx_split, 1)
        next_action = actor_target(next_observation, True) # TODO Remove True?

        critit_input = tf.concat([next_observation, next_action], 1)
        critic_output = critic_target(critit_input, True) # TODO Remove True?
        target = reward + discount * critic_output

        # update weights
        actor_loss_grad, critic_loss_grad = \
            get_gradients(actor, critic, observation, action, target)

        critic_optimizer.apply_gradients(zip(critic_loss_grad, critic.trainable_weights))
        actor_optimizer.apply_gradients(zip(actor_loss_grad, actor.trainable_weights))

        # update target weights
        update_target_weights(actor, critic, actor_target, critic_target)

        time_step = next_time_step

    running_reward.append(cumulative_reward)
    running_error.append(on_policy_error)

    print(f"{episode + 1}".ljust(4) \
          + f"reward: {cumulative_reward:.3f}".ljust(25) \
              + f"running {sum(running_reward) / len(running_reward):.3f}".ljust(25) \
                  + f"on-policy error: {on_policy_error:.3f}".ljust(25) \
        + f"running {sum(running_error) / len(running_error):.3f}".ljust(25) \
            + f"noise: {epsilon: .6f}".ljust(20))

    episode += 1

# =============================================================================
# === Test
test_env = DerivativeBookHedgeEnv(
    book, init_state, num_hedges, cost_scale, test_size)
benchmark = BlackScholesDeltaPolicy(book)
test_metric = TestMetric()

time_step = test_env.reset()
while not time_step.terminated:
    action_step = ActionStep(actor(time_step.observation))
    next_time_step = test_env.step(action_step)
    test_metric.load(time_step, action_step)
    time_step = next_time_step

test_metric_benchmark = TestMetric()
time_step = test_env.reset()
while not time_step.terminated:
    action_step = benchmark.action(time_step)
    next_time_step = test_env.step(action_step)
    test_metric_benchmark.load(time_step, action_step)
    time_step = next_time_step


plt.figure()

time = tf.linspace(0., book.maturity, num_hedges)

for metric in [test_metric, test_metric_benchmark]:
    mean, std, left_ci, right_ci = metric.result()

    plt.plot(time, mean)
    plt.plot(time, left_ci, "--", color="red")
    plt.plot(time, right_ci, "--", color="red")
    break

plt.show()

