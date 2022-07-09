import torchvision
import torch
from model import AutoEncoder
import torch.optim as optim
import torch.nn as nn
from dataloader import TinyImageNetDataset
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import random

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_folder = 'output'
weights_name = 'model.pt'
test_num = 10
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AutoEncoder().to(device)
model.load_state_dict(torch.load(weights_name), strict=False)

test_images_path = os.listdir('../../../../../../../thesis/tiny-imagenet-200/test/images')
test_images = []
for i in range(test_num):
    img = imageio.imread(os.path.join('../../../../../../../thesis/tiny-imagenet-200/test/images', test_images_path[i]))
    if len(img.shape) == 2:
      img = np.stack((img,)*3, axis=-1)

    img = cv2.resize(img, (128, 128))
    orig_img = img.copy()
    width = 128
    patch_size = random.randint(int(width*0.1), int(width*0.3))
    rand_x = random.randint(1, width - patch_size - 2)
    rand_y = random.randint(1, width - patch_size - 2)

    intensity_addition = random.randint(30, 50)
    img[rand_x: rand_x + patch_size, rand_y: rand_y + patch_size, :] = increase_brightness(img[rand_x: rand_x + patch_size, rand_y: rand_y + patch_size, :], intensity_addition)

    test_images.append((orig_img, img))

for i, data in enumerate(test_images):
    if i >= test_num:
        break
    orig_images, images = data
    images = torch.from_numpy(np.array(images).transpose((2, 1, 0))).to(device) / 255.0
    orig_images = torch.from_numpy(np.array(orig_images).transpose((2, 1, 0))).to(device) / 255.0

    # compute reconstructions
    outputs = model(images.unsqueeze(dim=0))

    # fig = plt.figure(figsize=(10, 7))
    # rows = 1
    # columns = 3
    #
    # fig.add_subplot(rows, columns, 1)
    # plt.plot(np.array(orig_images.cpu().detach().numpy() * 255.0).transpose(1, 2, 0).astype(np.uint8))
    # plt.axis('off')
    # plt.title("Ground Truth")
    #
    # fig.add_subplot(rows, columns, 2)
    # plt.plot(np.array(images.cpu().detach().numpy() * 255.0).transpose(1, 2, 0).astype(np.uint8))
    # plt.axis('off')
    # plt.title("Before Embedding")
    #
    # fig.add_subplot(rows, columns, 3)
    # plt.plot(np.array(outputs[0].cpu().detach().numpy() * 255.0).transpose(1, 2, 0).astype(np.uint8))
    # plt.axis('off')
    # plt.title("After Embedding")

    cv2.imwrite(os.path.join(output_folder, "sample_" + str(i).zfill(4) + "_gt.jpg"), np.array(orig_images.cpu().detach().numpy() * 255.0).transpose(1, 2, 0).astype(np.uint8))
    cv2.imwrite(os.path.join(output_folder, "sample_" + str(i).zfill(4) + "_bef.jpg"), np.array(images.cpu().detach().numpy() * 255.0).transpose(1, 2, 0).astype(np.uint8))
    cv2.imwrite(os.path.join(output_folder, "sample_" + str(i).zfill(4) + "_aft.jpg"), np.array(outputs[0].cpu().detach().numpy() * 255.0).transpose(1, 2, 0).astype(np.uint8))

    # plt.savefig(os.path.join(output_folder, "sample_" + str(i).zfill(4) + ".jpg"))

    # plt.imshow(  tensor_image.permute(1, 2, 0)  )
    # compute accumulated gradients

    # plt.imshow(np.array(outputs[0].permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8))
    # plt.show()
