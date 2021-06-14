import argparse
import time, copy, sys, os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from deeplite_torch_zoo.wrappers import get_data_splits_by_name


def train_model(output_path, model, dataloaders, criterion, optimizer, num_epochs=5):
    if not os.path.exists('models/' + str(output_path)):
        os.makedirs('models/' + str(output_path))
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                labels = labels.cuda()
                inputs = inputs.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")
                sys.stdout.flush()

            num_samples = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / num_samples
            epoch_acc = running_corrects.double() / num_samples
            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                test_loss = epoch_loss
                test_acc = epoch_acc

        print('\nTrain Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print(  'Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
        print()
        save_path = './models/' + str(output_path) + '/model_{}_epoch.pt'.format(epoch+1)
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def get_model(arch="resnet18", num_classes=100, pretrained=True, fp16=False, device="cuda"):
    if not pretrained:
        return eval(f"models.{arch}")(num_classes=num_classes).to(device) 

    weights = eval(f"models.{arch}")(pretrained=pretrained).state_dict()
    weights = {k: v for k, v in weights.items() if 'fc' not in k and 'classifier' not in k}
    model = eval(f"models.{arch}")(num_classes=num_classes)
    model.load_state_dict(weights, strict=False)
    if fp16:
        model = model.half()
    return model.to(device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--data-root", type=str, default="/neutrino/datasets/TinyImageNet/")
    parser.add_argument("--dataset-name", type=str, default="tinyimagenet")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=False)

    opt = parser.parse_args()

    dataloaders = get_data_splits_by_name(
        dataset_name=opt.dataset_name,
        data_root=opt.data_root,
        num_workers=0,
        batch_size=128,
        fp16=opt.fp16
    )
    model = get_model(arch=opt.arch, num_classes=opt.num_classes, pretrained=opt.pretrained, fp16=opt.fp16, device="cuda")

    #Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_model(f"{opt.dataset_name}/{opt.arch}", model, dataloaders, criterion, optimizer, num_epochs=opt.epochs)

