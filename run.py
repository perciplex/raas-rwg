import gym_raas
import gym
import numpy as np
best_weights = [[[-2.377215290124257, -6.584838561793895, -2.9259219971379324], [6.984564766333971, -9.594753869775147, -1.8737953249232997], [6.06378658149866, 3.0843715924465487, 1.630334868138612], [0.702983674192379, 8.25281420053158, 1.512670105911822]], [[3.6165917793791746, -5.64360367302934, 9.24114963935817, -8.004634102606525]]]

weights_mat = [np.array(w) for w in best_weights]


def get_action(input_vec):

    x = input_vec
    N_hidden_layers = len(weights_mat)

    for i, w in enumerate(weights_mat):
        x = np.dot(w, x)
        # Only apply activation function to intermediate layers
        x = np.tanh(x)

    x *= 2.0
    return x

env_name = 'raaspendulum-v0'
env = gym.make(env_name)


# Move it away from the bottom
n_steps = 50
DT = 0.05
max_torque = 1.2
w = 4.5
for t in range(n_steps):
    phase = np.sin(w * t * DT)
    if phase > 0:
        mult = 1.0
    else:
        mult = -1.0
    action = mult * max_torque
    #action = phase * max_torque
    observation, reward, done, info = env.step([action])

# Run ep
obs = env._get_obs()
score = 0
steps = 0
max_episode_steps = 200
show_ep = True
done = False
while not done:
    if show_ep:
        env.render()
        if steps % 10 == 0:
            print(f"step {steps}, score {score:.2f}")

    action = get_action(obs)
    obs, rew, done, info = env.step(action)
    score += rew
    steps += 1
    if steps >= max_episode_steps:
        done = True





#
