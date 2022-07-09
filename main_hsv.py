import torchvision
import torch
from model import AutoEncoder
import torch.optim as optim
import torch.nn as nn
from dataloader_hsv import TinyImageNetDataset

import matplotlib.pyplot as plt
import numpy as np

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AutoEncoder().to(device)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = TinyImageNetDataset("../../../../../../../thesis/tiny-imagenet-200", preload=False, mode='train', load_transform=True, transform=transform)
test_dataset = TinyImageNetDataset("../../../../../../../thesis/tiny-imagenet-200", preload=False, mode='test', load_transform=True, transform=transform)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()
smooth_criterion = nn.SmoothL1Loss()

epochs = 10

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, shuffle=False
)

for epoch in range(epochs):
    loss = 0
    for i, data in enumerate(train_loader, 0):
        orig_images, images, labels = data
        images = images.to(device)
        orig_images = orig_images.to(device)

        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(images)

        # compute training reconstruction loss
        train_loss = criterion(outputs, orig_images)
        # plt.imshow(  tensor_image.permute(1, 2, 0)  )
        # compute accumulated gradients
        train_loss.backward()

        # plt.imshow(np.array(outputs[0].permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8))
        # plt.show()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

        if i % 100 == 0:
            print("iteration : {}, loss = {:.6f}".format(i + 1, train_loss.item()))
            torch.save(model.state_dict(), "model_hsv.pt")
            #model.load_state_dict(torch.load("model.pt"), strict=False)

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
