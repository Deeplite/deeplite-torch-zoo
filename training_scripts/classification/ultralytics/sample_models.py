import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from deeplite_torch_zoo import get_model
from deeplite_torch_zoo.src.dnn_blocks.search_spaces.search_spaces import SEARCH_SPACES

from train import main, parse_opt


def main_sampling(args):

    rng = np.random.default_rng(args.seed)
    search_space = SEARCH_SPACES.get('cnn')
    block_type_dict = search_space.registry_dict
    block_type_keys = list(block_type_dict.keys())

    model = get_model(
        model_name='resnet18',
        dataset_name='flowers102',
        pretrained=False,
        num_classes=102,
    )

    df = defaultdict(list)
    for _ in range(args.trials):
        block_idxs = rng.integers(0, len(block_type_keys) - 1, size=5)
        df['arch'].append(block_idxs)

        model.layer1[0] = block_type_dict[block_type_keys[block_idxs[0]]](64, 64)
        model.layer1[1] = block_type_dict[block_type_keys[block_idxs[1]]](64, 64)
        model.layer2[1] = block_type_dict[block_type_keys[block_idxs[2]]](128, 128)
        model.layer3[1] = block_type_dict[block_type_keys[block_idxs[3]]](256, 256)
        model.layer4[1] = block_type_dict[block_type_keys[block_idxs[4]]](512, 512)

        opt = parse_opt()

        opt.data_root = args.data_root
        opt.dataset = 'flowers102'
        opt.workers = 16

        res = main(opt, model)
        df['top1'].append(res['top1'])
        df['top5'].append(res['top5'])

        pd.DataFrame(df).to_csv(f'sampling_results_seed{args.seed}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./', type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--trials', default=1000, type=int)
    args = parser.parse_args()
    main_sampling(args)
