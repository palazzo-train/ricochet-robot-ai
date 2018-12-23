
import numpy as np
import sys
import utils.plotting as plotting
from matplotlib import pyplot as plt
from matplotlib import pylab
import matplotlib.gridspec as gridspec

class Experiment(object):
    def __init__(self, env, agent, debug_print_interval = 0):
        
        self.env = env
        self.agent = agent
        
        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])
        
        self.fig = pylab.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2, 2)
        self.ax = pylab.subplot(gs[:, 0])
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        
        if hasattr(self.env, '_cliff'): # Hardcode to nicely display grid for cliffwalkingenv
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.set_xticks(np.arange(-.5, 12, 1), minor=True);
            self.ax.set_yticks(np.arange(-.5, 4, 1), minor=True);
            self.ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
            
        if hasattr(self.env, 'winds'): # Hardcode to nicely display grid for windygridworldenv
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.set_xticks(np.arange(-.5, 10, 1), minor=True);
            self.ax.set_yticks(np.arange(-.5, 7, 1), minor=True);
            self.ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        
        self.ax1 = pylab.subplot(gs[0, 1])
        self.ax1.yaxis.set_label_position("right")
        self.ax1.set_ylabel('Length')
        
        self.ax1.set_xlim(0, max(10, len(self.episode_length)+1))
        self.ax1.set_ylim(0, 51)
        
        self.ax2 = pylab.subplot(gs[1, 1])
        self.ax2.set_xlabel('Episode')
        self.ax2.yaxis.set_label_position("right")
        self.ax2.set_ylabel('Reward')
        self.ax2.set_xlim(0, max(10, len(self.episode_reward)+1))
        self.ax2.set_ylim(0, 2)
        
        self.line, = self.ax1.plot(range(len(self.episode_length)),self.episode_length)
        self.line2, = self.ax2.plot(range(len(self.episode_reward)),self.episode_reward)

        self.debug_print_interval = debug_print_interval
        
    def update_display_step(self):
        if not hasattr(self, 'imgplot'):
            self.imgplot = self.ax.imshow(self.env.render(mode='rgb_array'), interpolation='none', cmap='viridis')
        else:
            self.imgplot.set_data(self.env.render(mode='rgb_array'))
    
        self.fig.canvas.draw()
        
    def update_display_episode(self):  
        self.line.set_data(range(len(self.episode_length)),self.episode_length)
        self.ax1.set_xlim(0, max(10, len(self.episode_length)+1))
        self.ax1.set_ylim(0, max(self.episode_length)+1)
        
        self.line2.set_data(range(len(self.episode_reward)),self.episode_reward)
        self.ax2.set_xlim(0, max(10, len(self.episode_reward)+1))
        self.ax2.set_ylim(min(self.episode_reward)-1, max(self.episode_reward)+1)
        
        self.fig.canvas.draw()     
        
    def run_bandit(self, max_number_of_trials=1000, display_frequency=1):
        self.fig.clf()
        
        print("Distribution:", self.env.distribution, self.env.reward_parameters, flush = True)
        print("Optimal arm:", self.env.optimal_arm, flush = True)
        
        if self.env.distribution != "normal":
            plotting.plot_arm_rewards(self.env.reward_parameters)
        #else:
            #plotting.plot_arm_rewards(self.env.reward_parameters[0])
        
        stats = plotting.TimestepStats(
            cumulative_rewards=np.zeros(max_number_of_trials),
            regrets=np.zeros(max_number_of_trials))   
            
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        
        for trial in range(max_number_of_trials):
            action = self.agent.act()
            
            _ , reward, done, _ = self.env.step(action)       
            self.agent.feedback(action, reward)
            cumulative_reward += reward

            gap = self.env.compute_gap(action)
            if action != self.env.optimal_arm:
                cumulative_regret += gap

            stats.cumulative_rewards[trial] = cumulative_reward
            stats.regrets[trial] = cumulative_regret

        print("--------------------------------------------------", flush = True)
        print("Policy:", self.agent.name, "\nAverage Reward:", cumulative_reward / max_number_of_trials, \
                "\nAverage Regret:", cumulative_regret / max_number_of_trials, flush = True)
        print("Arm pulls:", self.agent.total_counts, flush = True)
         
        plotting.plot_reward_regret(stats)
        
    def run_agent(self, max_number_of_episodes=100, max_number_of_steps=100, interactive = False, display_frequency=1):

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):
            
            # initialize state
            state = self.env.reset()
            
            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length
            
            # repeat for each step of episode, until state is terminal
            while not done:
                
                # increase step counter - for display
                t += 1
                
                # choose action from state 
                action = self.agent.act(state)
                
                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)
                
                # state <- next state
                state = next_state
                
                R += reward # accumulate reward - for display
                
                # if interactive display, show update for each step
                if interactive:
                    self.update_display_step()
                
                if t > max_number_of_steps :
                    print( 'too many steps. Stopped')
                    break
            
            self.episode_length = np.append(self.episode_length,t) # keep episode length - for display
            self.episode_reward = np.append(self.episode_reward,R) # keep episode reward - for display 
            
            # if interactive display, show update for the episode
            if interactive:
                self.update_display_episode()
        
        # if not interactive display, show graph at the end
        if not interactive:
            self.fig.clf()
            stats = plotting.EpisodeStats(
                episode_lengths=self.episode_length,
                episode_rewards=self.episode_reward,
                episode_running_variance=np.zeros(max_number_of_episodes))
            plotting.plot_episode_stats(stats, display_frequency)
        
  
    def run_qlearning(self, max_number_of_episodes=100, interactive = False, display_frequency=1):

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):
            
            # initialize state
            state = self.env.reset()
            
            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length
            
            # repeat for each step of episode, until state is terminal
            while not done:
                
                t += 1 # increase step counter - for display
                
                # choose action from state using policy derived from Q
                action = self.agent.act(state)
                
                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # agent learn (Q-Learning update)
                self.agent.learn(state, action, reward, next_state, done)
                
                # state <- next state
                state = next_state
                
                R += reward # accumulate reward - for display
                
                # if interactive display, show update for each step
                if interactive:
                    self.update_display_step()
            
            self.episode_length = np.append(self.episode_length,t) # keep episode length - for display
            self.episode_reward = np.append(self.episode_reward,R) # keep episode reward - for display 

            if (self.debug_print_interval != 0 ):
                if episode_number % self.debug_print_interval == 0 :
                    print('episode : ' + str(episode_number))
                    self.agent.save_model()
            
            # if interactive display, show update for the episode
            if interactive:
                self.update_display_episode()
        
        # if not interactive display, show graph at the end
        if not interactive:
            self.fig.clf()
            stats = plotting.EpisodeStats(
                episode_lengths=self.episode_length,
                episode_rewards=self.episode_reward,
                episode_running_variance=np.zeros(max_number_of_episodes))
            plotting.plot_episode_stats(stats, display_frequency)
            
    def run_sarsa(self, max_number_of_episodes=100, interactive = False, display_frequency=1):

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):
            
            # initialize state
            state = self.env.reset()

            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length
            
            # choose action from state using policy derived from Q
            action = self.agent.act(state)
            
            # repeat for each step of episode, until state is terminal
            while not done:
                
                t += 1 # increase step counter - for display
                
                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)
                
                # choose next action from next state using policy derived from Q
                next_action = self.agent.act(next_state)
                
                # agent learn (SARSA update)
                self.agent.learn(state, action, reward, next_state, next_action)
                
                # state <- next state, action <- next_action
                state = next_state
                action = next_action

                R += reward # accumulate reward - for display
                
                # if interactive display, show update for each step
                if interactive:
                    self.update_display_step()
            
            self.episode_length = np.append(self.episode_length,t) # keep episode length - for display
            self.episode_reward = np.append(self.episode_reward,R) # keep episode reward - for display 

            print('episode : ' + str(episode))
            
            # if interactive display, show update for the episode
            if interactive:
                self.update_display_episode()
        
        # if not interactive display, show graph at the end
        if not interactive:
            self.fig.clf()
            stats = plotting.EpisodeStats(
                episode_lengths=self.episode_length,
                episode_rewards=self.episode_reward,
                episode_running_variance=np.zeros(max_number_of_episodes))
            plotting.plot_episode_stats(stats, display_frequency)

    def run_qlearning_enhanced():
        episodes = 1
        time_step = 10

        # initialize gym environment and the agent
        #env = gym.make('CartPole-v0')
        #agent = DQNAgent(env)
        # Iterate the game
        for e in range(episodes):
            # reset state in the beginning of each game
            state = env.reset()
            
            state = np.reshape(state, [1, 8])
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            for time_t in range(time_step):
                # turn this on if you want to render
                # env.render()
                # Decide action
                action = agent.act(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, 8])
                # Remember the previous state, action, reward, and done
                agent.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}"
                          .format(e, episodes, time_t))
                    break
                

            # train the agent with the experience of the episode
            agent.replay(32)
