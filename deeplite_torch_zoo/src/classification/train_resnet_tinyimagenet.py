import time, copy, sys, os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from deeplite_torch_zoo.wrappers import get_data_splits_by_name

def train_model(output_path, model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=5, scheduler=None):
    if not os.path.exists('models/'+str(output_path)):
        os.makedirs('models/'+str(output_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

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

#                 print( (i+1)*100. / len(dataloaders[phase]), "% Complete" )
                sys.stdout.flush()
                
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
            
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

        print('\nTrain Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print(  'Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print()
        save_path = './models/' + str(output_path) + '/model_{}_epoch.pt'.format(epoch+1)
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {}, Epoch: {}'.format(best_acc, best))


if __name__ == "__main__":
    data_dir = '/neutrino/datasets/TinyImageNet/'
    dataloaders = get_data_splits_by_name(
        dataset_name="tinyimagenet",
        data_root=data_dir,
        num_workers=0,
        batch_size=128,
        device="cuda",
    )
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    weights = models.resnet50(pretrained=True).state_dict()
    weights = {k: v for k, v in weights.items() if 'fc' not in k}

    model = models.resnet50(num_classes=100)
    model.load_state_dict(weights, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_model("64_pre_50", model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=20)


