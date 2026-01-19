from collections import deque
import numpy as np
import contextlib
import matplotlib.pyplot as plt
import sys

import torch


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1.0 / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception("Invalid action.")

        self.n_steps += 1
        done = self.n_steps >= self.max_steps

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.height = self.lake.shape[0]
        self.width = self.lake.shape[1]

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == "&")[0]] = 1.0

        self.absorbing_state = n_states - 1

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)

    def coord_to_state(self, x, y):
        return self.lake_flat[x * self.width + y]

    def state_to_coord(self, state):
        if state == self.absorbing_state:
            return self.width, self.height

        x = state % self.width
        y = state // self.width
        return x, y

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        # If going into absorbing state
        if state != self.absorbing_state:
            if self.lake_flat[state] == "#":
                if next_state == self.absorbing_state:
                    return 1.0
                else:
                    return 0.0

            if self.lake_flat[state] == "$":
                if next_state == self.absorbing_state:
                    return 1.0
                else:
                    return 0.0

        s_x, s_y = self.state_to_coord(state)
        ns_x, ns_y = self.state_to_coord(next_state)
        target_x, target_y = s_x, s_y

        # Try moving
        if action == 0:
            target_y -= 1

        elif action == 1:
            target_x -= 1

        elif action == 2:
            target_y += 1

        elif action == 3:
            target_x += 1

        adjacent_states = [
            (s_x + 1, s_y),
            (s_x - 1, s_y),
            (s_x, s_y + 1),
            (s_x, s_y - 1),
        ]

        valid_adjacent = []
        invalid_adjacent = []

        for x, y in adjacent_states:
            if 0 <= x < self.width and 0 <= y < self.height:
                valid_adjacent.append((x, y))
            else:
                invalid_adjacent.append((x, y))

        # Calculate transition probabilities
        prob = 0.0

        # If landed out of bounds
        if (target_x, target_y) in invalid_adjacent:
            if (ns_x, ns_y) == (s_x, s_y):
                prob += 0.9

        # If landed on valid tile
        else:
            if (ns_x, ns_y) == (target_x, target_y):
                prob += 0.9

        # Calculate slippage probabilities
        slippage = 0.1 / 4

        for adj_x, adj_y in invalid_adjacent:
            if (ns_x, ns_y) == (s_x, s_y):
                prob += slippage

        for adj_x, adj_y in valid_adjacent:
            if (ns_x, ns_y) == (adj_x, adj_y):
                prob += slippage

        return prob

    def r(self, next_state, state, action):
        if next_state == self.absorbing_state and state != self.absorbing_state:
            if self.lake_flat[state] == "#":
                return 0.0
            elif self.lake_flat[state] == "$":
                return 1.0

        return 0.0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = "@"

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ["^", "<", "_", ">"]

            print("Lake:")
            print(self.lake)

            print("Policy:")
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print("Value:")
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def play(env):
    actions = ["w", "a", "s", "d"]

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input("\nMove: ")
        if c not in actions:
            raise Exception("Invalid action")

        state, r, done = env.step(actions.index(c))

        env.render()
        print("Reward: {0}.".format(r))


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)

    for i in range(max_iterations):
        delta = 0

        for s in range(env.n_states):
            v = value[s]
            new_v = 0
            a = policy[s]

            for ns in range(env.n_states):
                p = env.p(ns, s, a)
                r = env.r(ns, s, a)
                new_v += p * (r + gamma * value[ns])

            value[s] = new_v
            delta = max(delta, abs(v - new_v))

        if delta < theta:
            break

    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    for s in range(env.n_states):
        q_values = np.zeros(env.n_actions, dtype=float)

        for a in range(env.n_actions):
            for ns in range(env.n_states):
                p = env.p(ns, s, a)
                r = env.r(ns, s, a)
                q_values[a] += p * (r + gamma * value[ns])

        best_action = np.argmax(q_values)
        policy[s] = best_action

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    value = np.zeros(env.n_states, dtype=float)

    for i in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        new_policy = policy_improvement(env, value, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy, value, i + 1 # q1: (i + ..) returns the number of iterations


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=float)

    for i in range(max_iterations):
        delta = 0

        for s in range(env.n_states):
            v = value[s]
            q_values = np.zeros(env.n_actions, dtype=float)

            for a in range(env.n_actions):
                for ns in range(env.n_states):
                    p = env.p(ns, s, a)
                    r = env.r(ns, s, a)
                    q_values[a] += p * (r + gamma * value[ns])

            value[s] = np.max(q_values)
            delta = max(delta, abs(v - value[s]))

        if delta < theta:
            break

    policy = policy_improvement(env, value, gamma)
    return policy, value, i + 1 # q1: (same utility as policy iter)


def select_greedy_action(q_row, epsilon, random_state):
    if random_state.rand() < epsilon:
        a = random_state.randint(len(q_row))
    else:
        max_q = np.max(q_row)
        best_actions = np.where(q_row == max_q)[0]
        a = random_state.choice(best_actions)

    return a


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    episode_returns_arr = np.zeros(max_episodes) # q2: stores discounted return per episode

    for i in range(max_episodes):
        s = env.reset()
        done = False

        episode_return = 0.0
        t = 0

        a = select_greedy_action(q[s], epsilon[i], random_state)

        while not done:
            ns, r, done = env.step(a)
            episode_return += (gamma ** t) * r
            t += 1


            if done:
                q[s, a] += eta[i] * (r - q[s, a])
                break

            na = select_greedy_action(q[ns], epsilon[i], random_state)
            q[s, a] += eta[i] * (r + gamma * q[ns, na] - q[s, a])

            s = ns
            a = na
        episode_returns_arr[i] = episode_return

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value, episode_returns_arr


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    episode_returns_arr = np.zeros(max_episodes) # q2: (same utility as sarsa)


    for i in range(max_episodes):
        s = env.reset()
        done = False

        episode_return = 0.0
        t = 0


        while not done:
            a = select_greedy_action(q[s], epsilon[i], random_state)
            ns, r, done = env.step(a)

            episode_return += (gamma**t) * r
            t += 1

            q[s, a] = q[s, a] + eta[i] * (r + gamma * np.max(q[ns]) - q[s, a])
            s = ns

        episode_returns_arr[i] = episode_return


    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value, episode_returns_arr


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    episode_returns_arr = np.zeros(max_episodes) # q2: (same util; q, sarsa)


    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)

        a = select_greedy_action(q, epsilon[i], random_state)
        done = False

        episode_return = 0.0
        t = 0

        while not done:
            ns_features, r, done = env.step(a)

            episode_return += (gamma**t) * r
            t += 1


            ns_q = ns_features.dot(theta)

            if done:
                delta = r - q[a]
                theta = theta + eta[i] * delta * features[a]
                break

            a_next = select_greedy_action(ns_q, epsilon[i], random_state)
            delta = r + gamma * ns_q[a_next] - q[a]
            theta = theta + eta[i] * delta * features[a]

            features = ns_features
            q = ns_q
            a = a_next

        episode_returns_arr[i] = episode_return

    return theta, episode_returns_arr


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    episode_returns_arr = np.zeros(max_episodes) # q2: (same util; q, sarsa, lin_q) 

    for i in range(max_episodes):
        features = env.reset()
        done = False

        episode_return = 0.0
        t = 0


        while not done:
            q = features.dot(theta)
            a = select_greedy_action(q, epsilon[i], random_state)

            ns_features, r, done = env.step(a)

            episode_return += (gamma**t) * r
            t += 1


            ns_q = ns_features.dot(theta)

            if done:
                delta = r - q[a]
                theta = theta + eta[i] * delta * features[a]
                break

            delta = r + gamma * ns_q.max() - q[a]
            theta = theta + eta[i] * delta * features[a]
            features = ns_features

        episode_returns_arr[i] = episode_return

    return theta, episode_returns_arr


class FrozenLakeImageWrapper:
    def __init__(self, env):
        self.env = env

        lake = self.env.lake

        self.n_actions = self.env.n_actions
        self.state_shape = (4, lake.shape[0], lake.shape[1])

        lake_image = [(lake == c).astype(float) for c in ["&", "#", "$"]]

        self.state_image = {
            self.env.absorbing_state: np.stack([np.zeros(lake.shape)] + lake_image)
        }
        for state in range(lake.size):
            agent_layer = (
                (np.arange(lake.size) == state).reshape(lake.shape).astype(float)
            )

            self.state_image[state] = np.stack([agent_layer] + lake_image)

    def encode_state(self, state):
        return self.state_image[state]

    def decode_policy(self, dqn):
        states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
        q = dqn(states).detach().numpy()  # torch.no_grad omitted to avoid import

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


class DeepQNetwork(torch.nn.Module):
    def __init__(
        self, env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed
    ):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.conv_layer = torch.nn.Conv2d(
            in_channels=env.state_shape[0],
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=1,
        )

        h = env.state_shape[1] - kernel_size + 1
        w = env.state_shape[2] - kernel_size + 1

        self.fc_layer = torch.nn.Linear(
            in_features=h * w * conv_out_channels, out_features=fc_out_features
        )
        self.output_layer = torch.nn.Linear(
            in_features=fc_out_features, out_features=env.n_actions
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)

        x = self.conv_layer(x)
        x = torch.nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        x = torch.nn.functional.relu(x)
        x = self.output_layer(x)
        return x

    def train_step(self, transitions, gamma, tdqn):
        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions])
        next_states = np.array([transition[3] for transition in transitions])
        dones = torch.tensor(
            [transition[4] for transition in transitions], dtype=torch.float32
        )

        q = self(states)
        q = q.gather(1, torch.Tensor(actions).view(len(transitions), 1).long())
        q = q.view(len(transitions))

        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)

        target = torch.tensor(rewards, dtype=torch.float32) + gamma * next_q

        loss = torch.nn.functional.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        self.buffer = deque(maxlen=buffer_size)
        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def draw(self, batch_size):
        indices = self.random_state.choice(
            len(self.buffer), size=batch_size, replace=False
        )
        return [self.buffer[i] for i in indices]


def deep_q_network_learning(
    env,
    max_episodes,
    learning_rate,
    gamma,
    epsilon,
    batch_size,
    target_update_frequency,
    buffer_size,
    kernel_size,
    conv_out_channels,
    fc_out_features,
    seed,
):
    random_state = np.random.RandomState(seed)
    replay_buffer = ReplayBuffer(buffer_size, random_state)

    dqn = DeepQNetwork(
        env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed=seed
    )
    tdqn = DeepQNetwork(
        env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed=seed
    )

    epsilon = np.linspace(epsilon, 0, max_episodes)

    episode_returns_arr = np.zeros(max_episodes)  # q2: store discounted return per episode

    for i in range(max_episodes):
        state = env.reset()

        episode_return = 0.0
        t = 0

        done = False
        while not done:
            if random_state.rand() < epsilon[i]:
                action = random_state.choice(env.n_actions)
            else:
                with torch.no_grad():
                    q = dqn(np.array([state]))[0].numpy()

                qmax = max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                action = random_state.choice(best)

            next_state, reward, done = env.step(action)

            episode_return += (gamma**t) * reward
            t += 1

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)

        episode_returns_arr[i] = episode_return

        if (i % target_update_frequency) == 0:
            tdqn.load_state_dict(dqn.state_dict())

    return dqn, episode_returns_arr

def main():
    seed = 0

    # Big lake
    #lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]

    # Small lake
    lake = [
        ["&", ".", ".", "."],
        [".", "#", ".", "#"],
        [".", ".", ".", "#"],
        ["#", ".", ".", "$"],
    ]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9

    print("# Model-based algorithms")

    print("")

    print("## Policy iteration")
    policy, value, iterations = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print("")

    print("## Value iteration")
    policy, value, iterations = value_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print("")

    print("# Model-free algorithms")
    max_episodes = 4000

    print("")

    print("## Sarsa")
    policy, value = sarsa(
        env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed
    )
    env.render(policy, value)

    print("")

    print("## Q-learning")
    policy, value = q_learning(
        env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed
    )
    env.render(policy, value)

    print("")

    linear_env = LinearWrapper(env)

    print("## Linear Sarsa")

    parameters = linear_sarsa(
        linear_env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed
    )
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print("")

    print("## Linear Q-learning")

    parameters = linear_q_learning(
        linear_env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed
    )
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print("")

    image_env = FrozenLakeImageWrapper(env)

    print("## Deep Q-network learning")

    dqn = deep_q_network_learning(
        image_env,
        max_episodes,
        learning_rate=0.001,
        gamma=gamma,
        epsilon=0.2,
        batch_size=32,
        target_update_frequency=4,
        buffer_size=256,
        kernel_size=3,
        conv_out_channels=4,
        fc_out_features=8,
        seed=4,
    )
    policy, value = image_env.decode_policy(dqn)
    image_env.render(policy, value)

# ===================================================================
# section 1.7 question 1 - 3
# ================================================================
# define lakes
b_lake = [ # big lake
            ['&', '.', '.', '.', '.', '.', '.', '.'],  
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', '.', '#', '.'],
            ['.', '#', '.', '.', '#', '.', '#', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '$']
        ]

s_lake = [ # small lake
    ["&", ".", ".", "."], 
    [".", "#", ".", "#"],
    [".", ".", ".", "#"],
    ["#", ".", ".", "$"],
]
# ================================================================


# ================================================================
# 1.
#=================================================================
def question_1(lake=b_lake):
    seed = 0
    print(f"1. big lake p_i vs v_i")

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9

    print("# Model-based algorithms")

    print("")

    print("## Policy iteration")
    policy, value, iterations = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)
    print(f"Iterations: {iterations}")

    print("")

    print("## Value iteration")
    policy, value, iterations = value_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)
    print(f"Iterations: {iterations}")

    print("")


# ================================================================
# 2.
#=================================================================
def question_2(lake=s_lake):
    print(f"2. small lake model-free algo returns experiment")
    seed = 0

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9
    max_episodes = 4000

    _, _, sarsa_returns = sarsa( # sarsa
        env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed
    )

    _, _, q_learning_returns = q_learning( #q-learning
        env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed
    )

    linear_env = LinearWrapper(env)
    _, linear_sarsa_returns = linear_sarsa( #linear sarsa
        linear_env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed
    )

    _, linear_q_learning_returns = linear_q_learning( #linear q learning control
        linear_env, max_episodes, eta=0.5, gamma=gamma, epsilon=0.5, seed=seed
    )

    image_env = FrozenLakeImageWrapper(env)
    _, dqn_returns = deep_q_network_learning( #deep q nework
        image_env,
        max_episodes,
        learning_rate=0.001,
        gamma=gamma,
        epsilon=0.2,
        batch_size=32,
        target_update_frequency=4,
        buffer_size=256,
        kernel_size=3,
        conv_out_channels=4,
        fc_out_features=8,
        seed=4,  # Using seed=4 as per assignment example
    )

    # q2: Print the sum of discounted returns for each algorithm
    print("\n--- Total Discounted Returns (Sum over all episodes) ---")
    print(f"Sarsa: {np.sum(sarsa_returns):.2f}")
    print(f"Q-Learning: {np.sum(q_learning_returns):.2f}")
    print(f"Linear Sarsa: {np.sum(linear_sarsa_returns):.2f}")
    print(f"Linear Q-Learning: {np.sum(linear_q_learning_returns):.2f}")
    print(f"Deep Q-Network: {np.sum(dqn_returns):.2f}")
    print("------------------------------------------------------\n")


    print("Plotting results...")

    # --- Plotting ---
    window = 20
    
    sarsa_ma = np.convolve(sarsa_returns, np.ones(window)/window, mode='valid')
    q_learning_ma = np.convolve(q_learning_returns, np.ones(window)/window, mode='valid')
    linear_sarsa_ma = np.convolve(linear_sarsa_returns, np.ones(window)/window, mode='valid')
    linear_q_learning_ma = np.convolve(linear_q_learning_returns, np.ones(window)/window, mode='valid')
    dqn_ma = np.convolve(dqn_returns, np.ones(window) / window, mode="valid")
    
    # The x-axis for the moving average starts after the first window
    x_axis = np.arange(window - 1, max_episodes)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(x_axis, sarsa_ma, label='Sarsa')
    plt.plot(x_axis, q_learning_ma, label='Q-Learning')
    plt.plot(x_axis, linear_sarsa_ma, label='Linear Sarsa')
    plt.plot(x_axis, linear_q_learning_ma, label='Linear Q-Learning')
    plt.plot(x_axis, dqn_ma, label="Deep Q-Network")
    
    plt.xlabel('Episode Number')
    plt.ylabel(f'Discounted Return (Moving Average over {window} episodes)')
    plt.title('Comparison of Model-Free Algorithms on Small Frozen Lake')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = "1_7_q2_plot.png"
    plt.savefig(plot_filename)
    plt.close()  # Free up memory
    print(f"Plot for Question 2 saved to '{plot_filename}'")


# ================================================================
# 3.
#=================================================================
# ===================================================================
# q3: Helper functions for Question 3: Hyperparameter Tuning
# These are modified versions of sarsa and q_learning to find
# the number of episodes required to converge to an optimal policy.
# ===================================================================

def sarsa_for_q3(
    env,
    max_episodes,
    eta,
    gamma,
    epsilon,
    seed=None,
    optimal_policy=None,
    convergence_patience=20,
):
    """SARSA algorithm modified to return episodes to convergence."""
    random_state = np.random.RandomState(seed)

    eta_decay = np.linspace(eta, 0, max_episodes)
    epsilon_decay = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    patience_counter = 0

    for i in range(max_episodes):
        s = env.reset()
        a = select_greedy_action(q[s], epsilon_decay[i], random_state)
        done = False

        while not done:
            ns, r, done = env.step(a)

            if done:
                q[s, a] += eta_decay[i] * (r - q[s, a])
                break

            na = select_greedy_action(q[ns], epsilon_decay[i], random_state)
            q[s, a] += eta_decay[i] * (r + gamma * q[ns, na] - q[s, a])

            s = ns
            a = na

        # q3: Check for convergence against the optimal policy
        if optimal_policy is not None:
            current_policy = q.argmax(axis=1)
            if np.array_equal(current_policy[:-1], optimal_policy[:-1]):
                patience_counter += 1
            else:
                patience_counter = 0  # Reset if policy changes

            if patience_counter >= convergence_patience:
                return i + 1  # Return episodes to converge

    return max_episodes  # Return max_episodes if not converged

def q_learning_for_q3(
    env,
    max_episodes,
    eta,
    gamma,
    epsilon,
    seed=None,
    optimal_policy=None,
    convergence_patience=20,
):
    """Q-learning algorithm modified to return episodes to convergence."""
    random_state = np.random.RandomState(seed)

    eta_decay = np.linspace(eta, 0, max_episodes)
    epsilon_decay = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    patience_counter = 0

    for i in range(max_episodes):
        s = env.reset()
        done = False

        while not done:
            a = select_greedy_action(q[s], epsilon_decay[i], random_state)
            ns, r, done = env.step(a)
            q[s, a] = q[s, a] + eta_decay[i] * (r + gamma * np.max(q[ns]) - q[s, a])
            s = ns

        if optimal_policy is not None:
            current_policy = q.argmax(axis=1)
            if np.array_equal(current_policy[:-1], optimal_policy[:-1]):
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= convergence_patience:
                return i + 1

    return max_episodes

def question_3(s_lake, b_lake):
    # q3: Implementation for question 3
    seed = 0
    gamma = 0.9
    max_episodes_small = 10000  # Max episodes for small lake search
    max_episodes_big = 20000  # Max episodes for big lake search

    # --- Part 1: Small Frozen Lake ---
    print("\n--- Question 3: Hyperparameter Tuning ---")
    print("\n--- Part 1: Small Frozen Lake ---")

    s_env = FrozenLake(s_lake, slip=0.1, max_steps=100, seed=seed)
    s_optimal_policy, s_optimal_value, _ = policy_iteration(
        s_env, gamma, theta=0.001, max_iterations=128
    )

    print("Optimal policy for small lake found via Policy Iteration:")
    s_env.render(s_optimal_policy, s_optimal_value)

    # q3: Define hyperparameter search space
    etas = [0.1, 0.3, 0.5, 0.7, 0.9]
    epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]

    # q3: Helper function to run the tuning process and report results
    def tune_and_report(algo_name, algo_func, env, optimal_policy, max_episodes):
        print(f"\n--- Tuning {algo_name} ---")
        best_params = {"eta": -1, "epsilon": -1}
        min_episodes = float("inf")

        for eta in etas:
            for epsilon in epsilons:
                # q3: Run multiple seeds for robustness and average the results
                episodes_needed_list = []
                for run_seed in range(3):  # 3 seeds for averaging
                    episodes_needed = algo_func(
                        env,
                        max_episodes,
                        eta,
                        gamma,
                        epsilon,
                        seed=seed + run_seed,
                        optimal_policy=optimal_policy,
                        convergence_patience=20,
                    )
                    episodes_needed_list.append(episodes_needed)

                avg_episodes = np.mean(episodes_needed_list)

                print(
                    f"  {algo_name} with eta={eta}, epsilon={epsilon}: ~{avg_episodes:.0f} episodes to converge."
                )

                if avg_episodes < min_episodes:
                    min_episodes = avg_episodes
                    best_params["eta"] = eta
                    best_params["epsilon"] = epsilon

        if min_episodes >= max_episodes:
            print(
                f"\nBest result for {algo_name}: Did not converge within {max_episodes} episodes."
            )
        else:
            print(f"\nBest result for {algo_name}:")
            print(f"  - Minimized episodes: ~{min_episodes:.0f}")
            print(f"  - Best eta: {best_params['eta']}")
            print(f"  - Best epsilon: {best_params['epsilon']}")
        return best_params, min_episodes

    # q3: Tune Sarsa and Q-learning for the small lake
    sarsa_best_params, sarsa_min_episodes = tune_and_report(
        "Sarsa", sarsa_for_q3, s_env, s_optimal_policy, max_episodes_small
    )
    q_learning_best_params, q_learning_min_episodes = tune_and_report(
        "Q-learning", q_learning_for_q3, s_env, s_optimal_policy, max_episodes_small
    )

    # --- Part 2: Big Frozen Lake ---
    print("\n\n--- Part 2: Big Frozen Lake ---")
    b_env = FrozenLake(b_lake, slip=0.1, max_steps=200, seed=seed)
    b_optimal_policy, b_optimal_value, _ = policy_iteration(
        b_env, gamma, theta=0.001, max_iterations=128
    )

    print("Optimal policy for big lake found via Policy Iteration:")
    b_env.render(b_optimal_policy, b_optimal_value)

    # q3: Tune Sarsa and Q-learning for the big lake
    tune_and_report("Sarsa", sarsa_for_q3, b_env, b_optimal_policy, max_episodes_big)
    tune_and_report(
        "Q-learning", q_learning_for_q3, b_env, b_optimal_policy, max_episodes_big
    )
    
class Tee:
    """A helper class to redirect stdout to both console and a file."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Flush to see output in real-time
    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == "__main__":
    # The assignment requires the output of the main function to be stored in output.txt
    # We will write the output for Q1, Q2, Q3 to both the console and a file.
    output_filename = "output.txt"
    print(f"Running questions 1, 2, and 3. Output will be printed to console and saved to '{output_filename}'...")
    print("Note: The plot for Question 2 will be displayed in a separate window. Please save it manually if needed.")

    original_stdout = sys.stdout  # Save a reference to the original standard output

    # The 'encoding' is set to 'utf-8' to handle special characters.
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Create a Tee object to write to both original stdout and the file
        tee = Tee(original_stdout, f)
        sys.stdout = tee

        try:
            # Call the functions whose output we want to capture
            question_1()
            question_2()
            # Note: plt.show() in question_2() is blocking and will pause execution here until the plot is closed.
            question_3(s_lake, b_lake)
        finally:
            # Restore stdout to its original state
            sys.stdout = original_stdout

    print(f"\nFinished. Output for questions 1, 2, and 3 also saved to '{output_filename}'.")