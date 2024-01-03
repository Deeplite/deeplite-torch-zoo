import argparse
import torch

from deeplite_torch_zoo import get_model
from deeplite_torch_zoo.utils.profiler import profile_ram, ram_report


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretraining-dataset', type=str, default='imagenet')
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--img-size', type=int, default=224, help='Image size (pixels)')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')

    return parser.parse_args()


def main(opt):
    model = get_model(
        model_name=opt.model,
        dataset_name=opt.pretraining_dataset,
        num_classes=opt.num_classes,
        pretrained=opt.pretrained,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img_size = (3, opt.img_size, opt.img_size)
    input_tensor = torch.randn(1, *img_size).to(device)

    # RAM Profiling - Get Maximum RAM Usage
    max_ram_usage = profile_ram(model, input_tensor)
    print(f"Maximum RAM Usage: {max_ram_usage} MB")

    # RAM Profiling - Detailed Layer-wise Report
    detailed_ram_data = profile_ram(model, input_tensor, detailed=True)
    ram_report(detailed_ram_data, verbose=opt.verbose, export=True, filename="ram_usage_report")

    return


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

