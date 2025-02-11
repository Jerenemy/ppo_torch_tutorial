from configs.config import load_config
from trainer import PPOTrainer

if __name__ == '__main__':
    config = load_config()
    trainer = PPOTrainer(config)
    trainer.train()
