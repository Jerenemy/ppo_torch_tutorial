import gymnasium as gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve

class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config['training']['env'])
        self.agent = Agent(
            n_actions=self.env.action_space.n, 
            batch_size=config['training']['batch_size'],
            alpha=config['training']['alpha'], 
            policy_clip=config['training']['policy_clip'], 
            n_epochs=config['training']['n_epochs'], 
            input_dims=self.env.observation_space.shape
        )
        self.figure_file = config['paths']['figure_file']
        self.n_games = config['training']['n_games']
        self.N = config['training']['N']

        # Tracking metrics
        self.best_score = -float('inf')
        self.score_history = []
        self.learn_iters = 0
        self.n_steps = 0

    def train(self):
        """Train the PPO agent."""
        for i in range(self.n_games):
            observation, _ = self.env.reset()
            done = False
            score = 0

            while not done:
                action, prob, val = self.agent.choose_action(observation)
                observation_, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.n_steps += 1
                score += reward

                self.agent.remember(observation, action, prob, val, reward, done)

                if self.n_steps % self.N == 0:
                    self.agent.learn()
                    self.learn_iters += 1

                observation = observation_

            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.agent.save_models()

            print(f'Episode {i}, Score: {score:.1f}, Avg Score: {avg_score:.1f}, '
                  f'Time Steps: {self.n_steps}, Learning Steps: {self.learn_iters}')

        self.save_results()

    def save_results(self):
        """Plot and save learning curve."""
        x = [i+1 for i in range(len(self.score_history))]
        plot_learning_curve(x, self.score_history, self.figure_file)

if __name__ == '__main__':
    pass