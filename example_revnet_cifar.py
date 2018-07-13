##
## Code adapted from https://github.com/tbung/pytorch-revnet
## commit : 7cfcd34fb07866338f5364058b424009e67fbd20
##

from datetime import datetime

import os
import sys
import argparse

from tqdm import tqdm
import torch
from torch.autograd import Variable
import torchvision
import numpy as np

from model import revnet
from data_tools.io import data_flow
from data_tools.data_augmentation import image_stack_random_transform


parser = argparse.ArgumentParser()
parser.add_argument("--model", metavar="NAME", default='revnet38', type=str,
                    help="what model to use")
parser.add_argument("--load", metavar="PATH",
                    help="load a previous model state")
parser.add_argument("-e", "--evaluate", action="store_true",
                    help="evaluate model on validation set")
parser.add_argument("--batch-size", default=128, type=int,
                    help="size of the mini-batches")
parser.add_argument("--epochs", default=200, type=int,
                    help="number of epochs")
parser.add_argument("--lr", default=0.1, type=float,
                    help="initial learning rate")
parser.add_argument("--clip", default=0, type=float,
                    help="maximal gradient norm")
parser.add_argument("--weight-decay", default=1e-4, type=float,
                    help="weight decay factor")
parser.add_argument("--stats", action="store_true",
                    help="record and plot some stats")


def main(use_gpu=False):
    args = parser.parse_args()
    
    # Set up model.
    model = getattr(revnet, args.model)()
    if use_gpu:
        model.cuda()
    if args.load is not None:
        load(model, args.load)
    
    # Set up experiment directory.
    exp_id = "cifar_{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}".format(model.name,
                                                          datetime.now())
    path = os.path.join("./experiments/", exp_id, "cmd.sh")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(' '.join(sys.argv))

    # Set up trainer.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr*10,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=50,
                                                gamma=0.1)

    # Prepare data.
    print("Prepairing data...")
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True)
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True)
    
    def preprocessor(data_augmentation=False):
        def f(batch):
        #b0, b1 = torch.utils.data.dataloader.default_collate(batch[0])
            b0, b1 = zip(*batch[0])
            b0 = list(b0)
            for i, image in enumerate(b0):
                b0[i] = np.array(image).transpose([2,1,0])
            
            if data_augmentation:
                b0 = image_stack_random_transform(b0,
                                                  width_shift_range=4,
                                                  height_shift_range=4,
                                                  fill_mode='reflect',
                                                  horizontal_flip=True)
        
            # Normalize
            mean =  np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
            stdev = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
            mean =  np.reshape(mean, (1, 3, 1, 1))
            stdev = np.reshape(stdev, (1, 3, 1, 1))
            b0 = np.subtract(b0, mean).astype(np.float32) / stdev
            
            # Package for pytorch
            b1 = np.array(b1, dtype=np.int64)
            b1 = torch.from_numpy(b1)
            b0 = torch.from_numpy(b0)
            
            return b0, b1
        return f

    trainloader = data_flow(data=[trainset],
                            batch_size=args.batch_size,
                            sample_random=True,
                            preprocessor=preprocessor(data_augmentation=True),
                            nb_io_workers=1,
                            nb_proc_workers=0)

    valloader = data_flow(data=[testset],
                          batch_size=args.batch_size,
                          sample_random=False,
                          preprocessor=preprocessor(),
                          nb_io_workers=1,
                          nb_proc_workers=0)

    if args.evaluate:
        print("\nEvaluating model...")
        acc = validate(model, valloader)
        print('Accuracy: {}%'.format(acc))
        return

    if args.stats:
        losses = []
        taccs = []
        vaccs = []

    print("\nTraining model...")
    best_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        loss, train_acc = train(epoch, model, criterion, optimizer,
                                trainloader, args.clip)
        val_acc = validate(model, valloader)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, exp_id)
        print('Accuracy: {}%'.format(val_acc))

        if args.stats:
            losses.append(loss)
            taccs.append(train_acc)
            vaccs.append(val_acc)

    save_checkpoint(model, exp_id)

    if args.stats:
        path = os.path.join("./experiments/", exp_id, "stats/{}.dat")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path.format('loss'), 'w') as f:
            for i in losses:
                f.write('{}\n'.format(i))

        with open(path.format('taccs'), 'w') as f:
            for i in taccs:
                f.write('{}\n'.format(i))

        with open(path.format('vaccs'), 'w') as f:
            for i in vaccs:
                f.write('{}\n'.format(i))

    return model


def train(epoch, model, criterion, optimizer, trainloader, clip):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    t = tqdm(trainloader, ascii=True, desc='{}'.format(epoch).rjust(3))
    for i, data in enumerate(t):
        inputs, labels = data

        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Free the memory used to store activations
        model.free()

        if clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        acc = 100 * correct / total

        t.set_postfix(loss='{:.3f}'.format(train_loss/(i+1)).ljust(3),
                      acc='{:2.1f}%'.format(acc).ljust(6))

    return train_loss, acc


def validate(model, valloader):
    correct = 0
    total = 0

    model.eval()

    for data in valloader:
        images, labels = data
        if use_gpu:
            images, labels = images.cuda(), labels.cuda()
        outputs = model(Variable(images))

        # Free the memory used to store activations
        model.free()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    acc = 100 * correct / total

    return acc


def load(model, path):
    model.load_state_dict(torch.load(path))


def save_checkpoint(model, exp_id):
    path = os.path.join(
        "experiments", exp_id, "checkpoints",
        "cifar_{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}.dat".format(model.name,
                                                         datetime.now()))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    main(use_gpu=use_gpu)
