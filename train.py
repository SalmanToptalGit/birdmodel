from trainers.trainer import Trainer
from cfgs.replicate_config import config

def main():
    trainer = Trainer()
    trainer.train_folds(config=config)

if __name__ == "__main__":
    main()
