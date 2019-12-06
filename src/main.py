print("hello")
#import sys
#import gym
#from dqn import DQNAgent
#from collections import deque
#import numpy as np
#
#
#env     = gym.make("MountainCar-v0")
#episodes  = 1000
#steps = 500
#print(env.observation_space)
#
#agent = DQNAgent(env.observation_space, env.action_space.n, eps_start=1.0, load=False)
#
#scores = []
#scores_buffer = deque(maxlen=100)
#avg_scores = []
#num_episodes = 2000
#for ep in range(num_episodes):
#    state = env.reset()
#    total_reward = 0
#    done = False
#    for step in range(steps):
#        action = agent.get_action(state)
#        next_state, reward, done, info = env.step(action)
#        agent.play(state, action, reward, next_state, done)
#        total_reward += reward  
#        state = next_state 
#        if done:
#            break
#
#    scores.append(total_reward)
#    scores_buffer.append(total_reward)
#    avg_scores.append(np.mean(scores_buffer))
#    if avg_scores[-1] >= np.max(avg_scores): agent.Q_network.save_model()
#    print("Episode: {}, Score: {}, Avg reward: {:.2f}".format(ep, scores[-1], avg_scores[-1]))
















# updateTargetNetwork = 1000
#dqn_agent = DQN(env=env)
#for episode in range(episodes):
#    cur_state = env.reset().reshape(1,2)
#    for step in range(steps):
#        action = dqn_agent.act(cur_state)
#        new_state, reward, done, _ = env.step(action)
#
#        # reward = reward if not done else -20
#        new_state = new_state.reshape(1,2)
#        dqn_agent.remember(cur_state, action, reward, new_state, done)
#
#        dqn_agent.replay()       # internally iterates default (prediction) model
#        dqn_agent.target_train() # iterates target model
#        env.render()
#
#        cur_state = new_state
#        if done:
#            break
#    if step >= 199:
#        print("Failed to complete in episode {}".format(episode))
#        if step % 10 == 0:
#            dqn_agent.save_model("episode-{}.model".format(episode))
#    else:
#        print("Completed in {} episodes".format(episode))
#        dqn_agent.save_model("success.model")
#        break
