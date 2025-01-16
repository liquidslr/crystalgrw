## Wrapper for crystalgrw.cli.train.main ##

import time
import argparse
from omegaconf import DictConfig, OmegaConf

from crystalgrw.cli.train import run_train


def main(cfg):
    run_train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--predict_property', default=False)
    parser.add_argument('--predict_property_class', default=False)
    parser.add_argument('--early_stop', type=int, default=300)

    args = parser.parse_args()

    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))

    cfg = OmegaConf.load(args.config_path)
    cfg.output_dir = args.output_path

    if args.predict_property is not None:
        cfg.model.predict_property = args.predict_property

    if args.predict_property_class is not None:
        cfg.model.predict_property_class = args.predict_property_class

    cfg.data = OmegaConf.load("./conf/data/" + cfg.data + ".yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))
    cfg.data.early_stopping_patience_epoch = args.early_stop

    main(cfg)
