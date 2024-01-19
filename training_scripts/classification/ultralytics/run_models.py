import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from deeplite_torch_zoo import get_model

from train import main, parse_opt


def main_sampling(args):
    df = defaultdict(list)
    with open(args.model_list_file, 'r') as file:
        model_list = [v.replace('\n', '') for v in file.readlines()]

    for model_name in model_list:
        try:
            model = get_model(
                model_name=model_name,
                dataset_name='flowers102',
                pretrained=False,
                num_classes=102,
            )

            df['model_name'].append(model_name)

            opt = parse_opt()

            opt.data_root = args.data_root
            opt.dataset = 'flowers102'
            opt.workers = 8

            res = main(opt, model)
            df['top1'].append(res['top1'])
            df['top5'].append(res['top5'])

            pd.DataFrame(df).to_csv(f'zoo_models_{args.model_list_file}.csv', index=False)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./', type=str)
    parser.add_argument('--model_list_file', type=str)
    args = parser.parse_args()
    main_sampling(args)
