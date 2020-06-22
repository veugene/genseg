# ============================================================
#
#   FIXME: SSL Mean Teacher
#   https://arxiv.org/pdf/1703.01780.pdf
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#  MedICAL Lab
#
# ============================================================

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import PIL
import copy

import dl_magic.dl_models as dl_models
from utils.pytorchtools import EarlyStopping, ModelCheckpoint, State_Variable
from utils.Datasets_Managment import NumPy_Dataset, Item_Normalize, Item_DeNormalize, PNG_Dataset

def model_func(channels, num_classes, device):

    model = dl_models.NoPoolNoBNASPP().to(device)

    return model

def train_dl(Dataset, model_filepath="SSL_Mean_Teacher_weights.best.pt"):
    # Hyper parameters
    epochs = int(1e3)
    num_classes = 5
    batch_size = 32
    learning_rate = 1e-1
    min_lr = 1e-12

    alpha_max = 0.99

    # Control variables
    global_step = 0

    [X_train, y_train, X_u_train, y_u_train, X_val, y_val, X_test, y_test] = Dataset

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    # Data
    # Training data generators

    # Define transforms
    train_transformations = transforms.Compose([  # transforms.ColorJitter(hue=.05, saturation=.05),
        # transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(35, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        # Item_Normalize(rgb=True)
        # transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    train_u_transformations = transforms.Compose([transforms.ToTensor()
                                                  # Item_Normalize(rgb=True)
                                                  #transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
                                                  ])

    val_transformations = transforms.Compose([transforms.ToTensor(),
                                              # Item_Normalize(rgb=True)
                                              # transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
                                              ])

    # Supervised train data generator
    train_dataset = PNG_Dataset(X_train, y_train, transforms=train_transformations)

    train_data_gen = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                num_workers=8,
                                shuffle=True)

    # TODO:Unsupervised train data generator
    train_u_dataset = PNG_Dataset(X_u_train, y_u_train, transforms=train_u_transformations)

    train_data_u_gen = DataLoader(dataset=train_u_dataset,
                                  batch_size=batch_size * 4,  # TODO: This improves the results
                                  num_workers=8,
                                  shuffle=False)

    # Validation data generator
    val_dataset = PNG_Dataset(X_val, y_val, transforms=val_transformations)

    val_data_gen = DataLoader(dataset=val_dataset,
                              batch_size=32,
                              num_workers=8,
                              shuffle=True)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model definition

    # model = model_inception_BN(input_channels=3, num_classes=num_classes).to(device)


    model_student = model_func(channels=3, num_classes=num_classes, device=device)
    model_teacher_ema = model_func(channels=3, num_classes=num_classes, device=device)


    for param in model_teacher_ema.parameters():
        param.detach_()

    # Model summary Keras style
    summary(model_student, (3, 32, 32))

    # Optimization parameters
    criterion_supervised = dice_loss
    criterion_consistency = softmax_mse_loss

    optimizer = torch.optim.SGD(model_student.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.5,
                                                           patience=10,
                                                           verbose=True,
                                                           threshold=0.001,
                                                           threshold_mode='rel',
                                                           cooldown=0,
                                                           min_lr=min_lr,
                                                           eps=1e-08)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=50, mode='max', verbose=True)
    checkpoint = ModelCheckpoint(checkpoint_fn=model_filepath, mode='max', verbose=True)

    # Save EMA model
    torch.save(model_teacher_ema.state_dict(), "EMA_" + model_filepath)

    # Training level variables
    # Logging

    loss_train = State_Variable()
    loss_supervised_train = State_Variable()
    loss_consistency_train = State_Variable()
    loss_val = State_Variable()
    loss_val_ema = State_Variable()

    dice_train = State_Variable()
    dice_val = State_Variable()
    dice_val_ema = State_Variable()

    # Tensorboard X

    writer = SummaryWriter(logdir='runs/MeanTeacher_' + str(len(y_train)))

    print('\n===== TRAINING =====\n')
    for epoch in range(epochs):

        # Epoch training
        # Epoch level variables

        loss_train.reset()
        loss_consistency_train.reset()
        loss_consistency_train.reset()
        loss_val.reset()
        loss_val_ema.reset()

        dice_train.reset()
        dice_val.reset()
        dice_val_ema.reset()

        ###################
        # train the model #
        ###################

        iter_number = 1000
        train_data_gen_iter = iter(train_data_gen)
        train_u_data_gen_iter = iter(train_data_u_gen)

        # TQDM progress bar definition, for visualization purposes
        pbar_train = tqdm(range(iter_number),
                          # total=len(train_data_gen),
                          total=iter_number,
                          unit=" iter",
                          leave=False,
                          desc='Train epoch ' + str(epoch + 1) + '/' + str(
                              epochs) + '   Loss: %.4f   Accuracy: %.3f  ' % (
                               loss_train.get_avg_value(), dice_train.get_avg_value()))

        # Prepare the model for train
        model_student.train()
        model_teacher_ema.train()

        for i in enumerate(pbar_train):

            # Epoch Training

            # Forward pass: compute predicted y by passing x to the model.

            try:
                X, y = next(train_data_gen_iter)
            except StopIteration:
                train_data_gen_iter = iter(train_data_gen)
                X, y = next(train_data_gen_iter)

            try:
                X_u, y_u = next(train_u_data_gen_iter)
            except StopIteration:
                train_u_data_gen_iter = iter(train_data_u_gen)
                X_u, y_u = next(train_u_data_gen_iter)

            X, y = X.to(device), y.to(device)
            X_u, y_u = X_u.to(device), y_u.to(device)

            y_pred = model_student(X)

            y_pred_u_student = model_student(X_u)
            y_pred_u_teacher = model_teacher_ema(X_u)

            # Compute and print loss.
            loss_supervised = criterion_supervised(y_pred, y)
            loss_consistency = 10*criterion_consistency(y_pred_u_student, y_pred_u_teacher) / num_classes

            loss = loss_supervised + loss_consistency

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

            update_ema_variables(model_student, model_teacher_ema, alpha_max, global_step)

            global_step += 1

            # Saving losses
            loss_train.update(loss.cpu().item())
            loss_supervised_train.update(loss_supervised.cpu().item())
            loss_consistency_train.update(loss_consistency.cpu().item())

            # Dice calculation
            dice_train.update(dice_loss(y_pred, y))

            # Update progress bar loss values
            pbar_train.set_description(
                'Train epoch ' + str(epoch + 1) + '/' + str(epochs) + '   Loss: %.4f   Dice: %.3f  ' % (
                loss_train.get_avg_value(), dice_train.get_avg_value()))

        # Saving train avg metrics to tensorboardX

        writer.add_scalar('losses/train_loss', loss_train.get_avg_value(), epoch)
        writer.add_scalar('losses/train_supervised_loss', loss_supervised_train.get_avg_value(), epoch)
        writer.add_scalar('losses/train_consistency_loss', loss_consistency_train.get_avg_value(), epoch)
        writer.add_scalar('accuracy/train_accuracy', dice_train.get_avg_value(), epoch)

        ######################
        # validate the model #
        ######################

        model_student.eval()  # prep model for evaluation

        # TQDM progress bar definition, for visualization purposes
        pbar_val = tqdm(iter(val_data_gen),
                        total=len(val_data_gen),
                        unit=" iter",
                        leave=False,
                        desc='Validation epoch ' + str(epoch + 1) + '/' + str(
                            epochs) + '   Loss: %.4f   Accuracy: %.3f  ' % (
                             loss_val.get_avg_value(), dice_val.get_avg_value()))

        for i, batch in enumerate(pbar_val):
            # Epoch Training

            # Forward pass: compute predicted y by passing x to the model.
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_pred = model_student(X)
            # Compute and print loss.
            loss = criterion_supervised(y_pred, y)
            # record validation loss

            # Saving losses
            loss_val.update(loss.cpu().item())

            # Dice calculation
            dice_val.update(dice_loss(y_pred, y))

            # Update progress bar loss values
            pbar_val.set_description(
                'Validation epoch ' + str(epoch + 1) + '/' + str(epochs) + '   Loss: %.4f   Accuracy: %.3f  ' % (
                loss_val.get_avg_value(), dice_val.get_avg_value()))

        # Saving val avg metrics on TensorboardX
        # writer.add_scalar('losses/val_loss', loss_val.get_avg_value(), epoch)
        # writer.add_scalar('accuracy/val_accuracy', acc_val.get_avg_value(), epoch)

        ##########################
        # validate the model ema #
        ##########################

        model_teacher_ema.eval()  # prep model for evaluation

        # TQDM progress bar definition, for visualization purposes
        pbar_val = tqdm(iter(val_data_gen),
                        total=len(val_data_gen),
                        unit=" iter",
                        leave=False,
                        desc='Validation epoch ' + str(epoch + 1) + '/' + str(
                            epochs) + '   Loss: %.4f   Accuracy: %.3f  ' % (
                                 loss_val.get_avg_value(), dice_val.get_avg_value()))

        for i, batch in enumerate(pbar_val):
            # Epoch Training

            # Forward pass: compute predicted y by passing x to the model.
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_pred = model_teacher_ema(X)
            # Compute and print loss.
            loss = criterion_supervised(y_pred, y)
            # record validation loss

            # Saving losses
            loss_val_ema.update(loss.cpu().item())

            # Dice calculation
            dice_val_ema.update(dice_loss(y_pred, y))

            # Update progress bar loss values
            pbar_val.set_description(
                'Validation epoch ' + str(epoch + 1) + '/' + str(epochs) + '   Loss: %.4f   Dice: %.3f  ' % (
                    loss_val_ema.get_avg_value(), dice_val_ema.get_avg_value()))

        # Saving val avg metrics on TensorboardX student and teacher models in the same graph

        writer.add_scalars('val_loss',
                           {'student_model': loss_val.get_avg_value(),
                            'teacher_ema_model': loss_val_ema.get_avg_value()},
                           epoch)
        writer.add_scalars('val_accuracy',
                           {'student_model': dice_val.get_avg_value(),
                            'teacher_ema_model': dice_val_ema.get_avg_value()},
                           epoch)

        # Saving val avg metrics on TensorboardX
        # writer.add_scalar('losses/val_loss_ema', loss_val_ema.get_avg_value(), epoch)
        # writer.add_scalar('accuracy/val_accuracy_ema', acc_val_ema.get_avg_value(), epoch)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(dice_val.get_avg_value())
        checkpoint(dice_val.get_avg_value(), model_student)

        # Reduce lr on plateau
        scheduler.step(dice_val.get_avg_value())

        if early_stopping.early_stop:
            print("Early stopping")
            break


# This fuction is used to update the Teacher model using the weights exp moving avg
def update_ema_variables(model_student, model_teacher, alpha_max, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha_max)
    for ema_param, param in zip(model_teacher.parameters(), model_student.parameters()):
        ema_param.data.mul_(alpha_max).add_(1 - alpha_max, param.data)



def dice_loss(input, target):
    # eps = 0.0001
    eps = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)

    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)

    return - dice

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=True)


def test_dl(Dataset, model, model_filepath="SSL_Mean_Teacher_weights.best.pt"):
    # Hyper parameters

    num_classes = 5
    batch_size = 1

    [X_train, y_train, X_u_train, y_u_train, val_set, val_set_GT, test_set, test_set_GT] = Dataset
    cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
    cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

    # Data
    # Test data generators
    test_transformations = transforms.Compose([transforms.ToTensor()
                                               # Item_Normalize(rgb=True)
                                               # transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
                                               ])

    test_dataset = PNG_Dataset(test_set, test_set_GT, transforms=test_transformations)

    test_data_gen = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               num_workers=4,
                               shuffle=False)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model definition

    model = model_func(channels=3, num_classes=num_classes, device=device)

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(model_filepath))

    # Model summary Keras style
    # summary(model, (1, 128, 128))

    # Optimization parameters
    criterion = torch.nn.CrossEntropyLoss()

    print('\n===== TESTING =====\n')

    # Epoch level variables

    loss_test = State_Variable()
    dice_test = State_Variable()

    # Variables to save predictions and labels
    y_pred_list = []
    y_list = []

    ######################
    # testing the model #
    ######################

    model.eval()  # prep model for evaluation

    # TQDM progress bar definition, for visualization purposes
    pbar_test = tqdm(iter(test_data_gen),
                     total=len(test_data_gen),
                     unit=" iter",
                     leave=False,
                     desc='Testing   Loss: %.4f   Accuracy: %.3f  ' % (
                     loss_test.get_avg_value(), dice_test.get_avg_value()))

    for i, batch in enumerate(pbar_test):
        # Epoch Training

        # Forward pass: compute predicted y by passing x to the model.
        X, y = batch
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        # Compute and print loss.
        loss = criterion(y_pred, y)
        # record validation loss

        # Saving loss
        loss_test.update(loss.cpu().item())

        #Dice calculation
        dice_test.update(dice_loss(y_pred, y))

        # Update progress bar loss values
        pbar_test.set_description(
            'Testing   Loss: %.4f   Dice: %.3f  ' % (loss_test.get_avg_value(), dice_test.get_avg_value()))

        # Save predictions and labels
        y_pred_list.append(y_pred.cpu().detach().numpy()[0])
        y_list.append(y.cpu().detach().numpy()[0])

    y_pred_list = np.asarray(y_pred_list)
    y_list = np.asarray(y_list)

    print('Accuracy: %.4f' % (dice_test.get_avg_value()))

    return [y_list, y_pred_list]


def test_dl_ema(Dataset, model, model_filepath="EMA_SSL_Mean_Teacher_weights.best.pt"):
    # Hyper parameters

    num_classes = 10
    batch_size = 32

    [X_train, y_train, X_u_train, y_u_train, val_set, val_set_GT, test_set, test_set_GT] = Dataset
    cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
    cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

    # Data
    # Test data generators
    test_transformations = transforms.Compose([transforms.ToTensor()
                                               # Item_Normalize(rgb=True)
                                               # transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
                                               ])

    test_dataset = PNG_Dataset(test_set, test_set_GT, transforms=test_transformations)

    test_data_gen = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               num_workers=4,
                               shuffle=False)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model definition

    model_func(channels=3, num_classes=num_classes, device=device)

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(model_filepath))

    # Model summary Keras style
    # summary(model, (1, 128, 128))

    # Optimization parameters
    criterion = torch.nn.CrossEntropyLoss()

    print('\n===== TESTING =====\n')

    # Epoch level variables

    loss_test = State_Variable()
    dice_test = State_Variable()

    # Variables to save predictions and labels
    y_pred_list = []
    y_list = []

    ######################
    # testing the model #
    ######################

    # model.eval()  # prep model for evaluation
    model.train()
    # TQDM progress bar definition, for visualization purposes
    pbar_test = tqdm(iter(test_data_gen),
                     total=len(test_data_gen),
                     unit=" iter",
                     leave=False,
                     desc='Testing   Loss: %.4f   Accuracy: %.3f  ' % (
                         loss_test.get_avg_value(), dice_test.get_avg_value()))

    for i, batch in enumerate(pbar_test):
        # Epoch Training

        # Forward pass: compute predicted y by passing x to the model.
        X, y = batch
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        # Compute and print loss.
        loss = criterion(y_pred, y)
        # record validation loss

        # Saving loss
        loss_test.update(loss.cpu().item())

        # Dice calculation

        dice_test.update(dice_loss(y_pred, y))

        # Update progress bar loss values
        pbar_test.set_description(
            'Testing   Loss: %.4f   Dice: %.3f  ' % (loss_test.get_avg_value(), dice_test.get_avg_value()))

        # Save predictions and labels
        y_pred_list.append(y_pred.cpu().detach().numpy()[0])
        y_list.append(y.cpu().detach().numpy()[0])

    y_pred_list = np.asarray(y_pred_list)
    y_list = np.asarray(y_list)

    print('Accuracy: %.4f' % (dice_test.get_avg_value()))

    return [y_list, y_pred_list]