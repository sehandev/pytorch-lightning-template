# PIP
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

# Custom
from train import train


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('--- [ Config List ] ---')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.common.seed)

    train(cfg)


if __name__ == '__main__':
    main()
