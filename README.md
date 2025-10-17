# SARSA Learning Algorithm
## AIM
Implementation of SARSA Learning Algorithm by integrating temporal differencing to find optimal policy for the given environment.

## PROBLEM STATEMENT
The given environment is the frozen lake environment where the agent must navigate from initial state to goal state avoiding holes. Various algorithms such as Value Iteration, First Visit Monte Carlo and SARSA algorithms are used to find the optimal policy for this environment. Compare these algorithms and identify the best algorithm.

## SARSA LEARNING ALGORITHM
# Step 1: 
Initial the required variables needed for the algorithm such as number of states, number of actions, lists to keep track of policies updated and the action value function.
# Step 2:
Define the select_action function which decides whether to explore or exploit and chooses an action according to the decision. 
# Step 3: 
Generate multiple learning rate and epsilon values you use for the algorithm.
# Step 4:
Iterate through episodes, compute TD target and TD error. Subsitute in the equation to find the Action Value function. Update policy choosing action with maximum value function.
# Step 5:
Return the results derived.

## SARSA LEARNING FUNCTION
### Name: Cynthia Mehul J
### Register Number: 212223240020
```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action=lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha,
                           min_alpha,
                           alpha_decay_ratio,
                           n_episodes)

    epsilons = decay_schedule(init_epsilon,
                              min_epsilon,
                              epsilon_decay_ratio,
                              n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilons[e])

      while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = select_action(next_state, Q, epsilons[e])
        td_target=reward+gamma*Q[next_state][next_action]*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]+=alphas[e]*td_error
        state, action = next_state, next_action
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
      V=np.max(Q,axis=1)
      pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:

Optimal value function

<img width="335" height="596" alt="image" src="https://github.com/user-attachments/assets/d4f9c021-77c7-4594-8516-0c8ac03b3ce8" />

Policy and Success rate for the optimal policy.

<img width="554" height="110" alt="image" src="https://github.com/user-attachments/assets/d5c110b1-be9e-4681-bda3-8e40bd9c7c21" />

SARSA value function

<img width="1091" height="785" alt="image" src="https://github.com/user-attachments/assets/a5ad5780-492b-4c50-a397-42d79cd99ebf" />

Policy and Success rate for SARSA

<img width="746" height="140" alt="image" src="https://github.com/user-attachments/assets/e90766aa-64a7-4ecc-86aa-a1990ecd5471" />

Comparison of the state value functions of Monte Carlo method and SARSA learning.

Monte Carlo
<img width="1784" height="661" alt="image" src="https://github.com/user-attachments/assets/70f7af36-9da1-42f7-83ee-02d126ae8703" />

SARSA 

<img width="1778" height="660" alt="image" src="https://github.com/user-attachments/assets/bb507889-9df6-42a7-9db6-d3209ff417bc" />

## RESULT:
Therefore, SARSA learning algorithm is implemented successfully to find optimal policy for the given environment.
