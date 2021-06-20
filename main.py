from config import config
from trainer import Trainer
from models.model import Model

if __name__ == '__main__':
    cfg = config()
    model = Model(cfg)
    trainer = Trainer(cfg, model)
    trainer.train()
