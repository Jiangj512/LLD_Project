import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=(axis + 1)).copy()
    return image


def random_rotate(image):
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, order=0, reshape=False, axes=(1, 2))
    return image


def random_zoom(image):
    img_shape = image.shape
    ratio = np.random.uniform(0.7, 1.4)
    # print(ratio)
    if ratio < 1:
        img = image.copy()
    image = zoom(image, (1, ratio, ratio), order=3)
    new_img_shape = image.shape
    # print(new_img_shape)
    if new_img_shape > img_shape:
        x = int((new_img_shape[1] - img_shape[1]) / 2)
        y = int((new_img_shape[1] - img_shape[1]) / 2 + 0.5)
        # print(x, y)
        image = image[:, x:(new_img_shape[1] - y), x:(new_img_shape[1] - y)]
    elif new_img_shape < img_shape:
        # new_label = np.zeros_like(lab)
        bais = sorted((img[0, 0, 0], img[0, img_shape[1] - 1, 0],
                       img[0, 0, img_shape[1] - 1], img[0, img_shape[1] - 1, img_shape[1] - 1]))
        new_image = np.zeros_like(img) + bais[1]
        x = int((img_shape[1] - new_img_shape[1]) / 2)
        y = int((img_shape[1] - new_img_shape[1]) / 2 + 0.5)
        # print(x, y)
        new_image[:, x:(img_shape[1] - y), x:(img_shape[1] - y)] = image
        return new_image

    return image


def random_brightness(image):
    ratio = np.random.uniform(0.7, 1.3)
    image = image * ratio
    return image


def random_gamma(image, epsilon=1e-7):
    img_mean = image.mean()
    img_sd = image.std()
    ratio = 1.3
    img_min = image.min()
    img_rnge = image.max() - img_min
    image = np.power(((image - img_min) / float(img_rnge + epsilon)), ratio) * img_rnge + img_min
    image = image - image.mean()
    image = image / (image.std() + 1e-8) * img_sd
    image = image + img_mean
    return image


def random_contrast(image):
    ratio = 1.2
    img_max = np.max(image)
    img_min = np.min(image)
    img_mean = np.mean(image)
    image = (image - img_mean) * ratio + img_mean
    image[image < img_min] = img_min
    image[image > img_max] = img_max
    return image


def random_gaussian_noise(image):
    ratio = np.random.uniform(0, 0.1)
    image = image + np.random.normal(0.0, ratio, size=image.shape)
    return image


def random_gaussian_blur(image):
    ratio = np.random.uniform(0.5, 1.5)
    for c in range(image.shape[0]):
        image[c] = gaussian_filter(image[c], ratio, order=0)
    return image


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        _, x, y = image.shape

        if random.random() < 0.5:
            image = random_rot_flip(image)
        if random.random() < 0.25:
            image = random_zoom(image)
        if random.random() < 0.25:
            image = random_rotate(image)

        if random.random() < 0.15:
            image = random_gaussian_noise(image)
        if random.random() < 0.2:
            image = random_gaussian_blur(image)
        if random.random() < 0.15:
            image = random_brightness(image)
        if random.random() < 0.2:
            image = random_contrast(image)
        if random.random() < 0.2:
            image = random_gamma(image)

        #         print(image.shape,label.shape)
        image = torch.from_numpy(image.astype(np.float32))

        sample = {'image': image, 'label': label}
        return sample


class LLD_dataset(Dataset):
    def __init__(self, base_dir, lld_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(lld_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train_net1":
            slice_name = self.sample_list[idx].strip('\n')
            # print(slice_name)
            image_path = os.path.join(self.data_dir, 'train', slice_name)

            img_data = np.load(image_path)

            image = img_data

            label = int((slice_name.split('_')[-1]).split('.')[0])

        if self.split == "val_net1":
            slice_name = self.sample_list[idx].strip('\n')
            # print(slice_name)
            image_path = os.path.join(self.data_dir, 'train', slice_name)

            img_data = np.load(image_path)

            image = img_data

            image = torch.from_numpy(image.astype(np.float32))
            #
            label = int((slice_name.split('_')[-1]).split('.')[0])

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
