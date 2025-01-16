## Wrapper for crystalgrw.cli.compute_metrics.main ##

import argparse
from crystalgrw.cli.compute_metrics import run_compute_metrics

def main(cfg):
    run_compute_metrics(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--classifier_path')
    parser.add_argument('--type', default='gen')
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    args = parser.parse_args()
    main(args)