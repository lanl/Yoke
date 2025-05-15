import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from yoke.models.mnist_model import Net

# To run: (myenv) C:\Users\364235\yoke>python -m harnesses.mnist_surrogate.mnist_harness

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    train function trains the model for one epoch:
    '''
    # Set model to training mode
    model.train()

    # Iterate over batches of data in the dataset
    for batch_idx, (data, target) in enumerate(train_loader):

        # Move data and target tensors to the specified device (CPU/GPU)
        data, target = data.to(device), target.to(device)

        # Zero the gradients of the optimizer
        optimizer.zero_grad()

        # Perfom the forward pass, assign predictions to the output variable
        output = model(data)

        # Calculates the loss value of the output, assigns it to loss variable
        loss = F.nll_loss(output, target)

        # Backpropogate the gradients of the loss function with respect to the model parameters
        loss.backward()

        # Update the weights based on the computed gradients
        optimizer.step()

        # If it's time to log the training status, prints the status.
        # If the dry_run flag is set, breaks the loop after the first batch
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    '''
    test function evaluates the model on the test dataset:
    '''
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for the test loss and number of correct predictions
    test_loss = 0
    correct = 0

    # Disable gradient computation
    with torch.no_grad():

        # Iterate over the batches of data
        for data, target in test_loader:

            #Move data and target tensors to the specified device
            data, target = data.to(device), target.to(device)

            # Perfom the forward pass, assign predictions to the output variable
            output = model(data)

            # sum up loss for all smaples in the batch
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # make predictions
            pred = output.argmax(dim=1, keepdim=True)

            # count number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Compute and print the average test loss and accuracy
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def parse_args():
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False, help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--conv1', type=int, default=32, help='Size of the first convolutional layer (default: 32)')
    parser.add_argument('--conv2', type=int, default=64, help='Size of the second convolutional layer (default: 64)')
    parser.add_argument('--conv3', type=int, default=128, help='Size of the third convolutional layer (default: 128)')
    parser.add_argument('--conv4', type=int, default=128, help='Size of the fourth convolutional layer (default: 128)')
    parser.add_argument('--config-file', type=str, help='path to configuration file')

    args = parser.parse_args()

    # If config file is provided, override the arguments
    if args.config_file:
        with open(args.config_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                key = lines[i].strip().lstrip('--')
                value = lines[i+1].strip()
                if hasattr(args, key):
                    attr_type = type(getattr(args, key))
                    setattr(args, key, attr_type(value))

    return args

def main():
    '''
    main function sets up the training and testing process:
    '''
    # Retrieving training settings
    args = parse_args()

    #Determining if CUDA or macOS GPU (MPS) is available.
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # Setting the random seed
    torch.manual_seed(args.seed)

    # Deciding which device to perform the training on
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Creating dictionaries that will be used to pass keyword arguments to Dataloaders for training and test datasets
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Normalizing the data to optimize performance of training
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Loading the training dataset
    dataset1 = datasets.MNIST('../data/MNIST', train=True, download=True,
                       transform=transform)
    
    # Loading the test dataset
    dataset2 = datasets.MNIST('../data/MNIST', train=False,
                       transform=transform)
    
    # Creating data loaders for the training and test datasets
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Creating model and preparing it for training or evaluation on the specified device
    model = Net(conv1_size=args.conv1, conv2_size=args.conv2, conv3_size=args.conv3, conv4_size=args.conv4).to(device)

    # Creating an Adadelta optimizer, which will be used to update the weights of the neural network during training
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Creating a learning rate scheduler, whcich can improve training performance by dynamically adjusting the learning rate
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Iterating through each epoch: training the model, testing the model, then updating the learning rate
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # If the --save-model flag was set, saves the trained model's state dictionary to a file named "mnist_cnn.pt". This contains
    # all the learnable parameters of the model, such as weights and biases, which can be loaded back into a model later
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
