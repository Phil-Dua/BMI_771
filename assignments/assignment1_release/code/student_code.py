import math
import random
import glob
import os
import numpy as np

import cv2
import numbers
import collections
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import torch
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F

from utils import resize_image, load_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Scale(320),
        >>>     RandomSizedCrop(224),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        repr_str = ""
        for t in self.transforms:
            repr_str += t.__repr__() + "\n"
        return repr_str


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly
    (with a probability of 0.5).
    """

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be flipped.

        Returns:
            numpy array: Randomly flipped image
        """
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            return img
        return img

    def __repr__(self):
        return "Random Horizontal Flip"


#################################################################################
# Sample implementation. No need to modify
#################################################################################

class RandomSizedCrop(object):
    """Crop the given numpy array to random area and aspect ratio.

    This class will crop a random region with in an input image. The target area
    / aspect ratio (=width/height) of the region is first sampled from a uniform
    distribution. A region satisfying the area / aspect ratio is then sampled
    and cropped. This crop is finally resized to a fixed given size. This is
    widely used as data augmentation for training image classification models.

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
        area_range (list of int): range of areas to sample from
        ratio_range (list of int): range of aspect ratios to sample from
        num_trials (int): number of sampling trials
    """

    def __init__(
        self,
        size,
        interpolations=_DEFAULT_INTERPOLATIONS,
        area_range=(0.25, 1.0),
        ratio_range=(0.8, 1.2),
        num_trials=10,
    ):
        self.size = size
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations
        self.num_trials = int(num_trials)
        self.area_range = area_range
        self.ratio_range = ratio_range

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        for attempt in range(self.num_trials):

            # sample target area / aspect ratio from area range and ratio range
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
            aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

            # compute the width and height
            # crop the image and resize to output size
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            # crop the image
            if w <= img.shape[1] and h <= img.shape[0]:
                x1 = random.randint(0, img.shape[1] - w)
                y1 = random.randint(0, img.shape[0] - h)

                img = img[y1 : y1 + h, x1 : x1 + w]
                if isinstance(self.size, int):
                    img = resize_image(
                        img, (self.size, self.size), interpolation=interpolation
                    )
                else:
                    img = resize_image(img, self.size, interpolation=interpolation)

                return img

        # Fall back
        if isinstance(self.size, int):
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            # with a square sized output, the default is to crop the patch in the center
            # (after all trials fail)
            h, w = img.shape[0], img.shape[1]
            th, tw = self.size, self.size
            x1 = int(round((w - tw) / 2.0))
            y1 = int(round((h - th) / 2.0))
            img = img[y1 : y1 + th, x1 : x1 + tw]
            return img
        else:
            # with a pre-specified output size, the default crop is the image itself
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return (
            "Random Crop"
            + "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}; Ratio {:.2f} - {:.2f}]".format(
                target_size[0],
                target_size[1],
                self.area_range[0],
                self.area_range[1],
                self.ratio_range[0],
                self.ratio_range[1],
            )
        )


#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
    """Rescale the input numpy array to the given size.

    This class will resize an input image based on its shortest side.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width)

        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
            Pass None during testing: always use CV2.INTER_LINEAR
    """

    def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2
        )
        self.size = size
        # use bilinear if interpolation is not specified
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be scaled.

        Returns:
            numpy array: Rescaled image
        """
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        # scale the image (code filled)
        if isinstance(self.size, int):
            height, width = img.shape[:2]

            # Preserve the aspect ratio while making the shorter side self.size.
            if width < height:
                target_width = self.size
                target_height = int(round(height * self.size / width))
            else:
                target_height = self.size
                target_width = int(round(width * self.size / height))

            return resize_image(
                img, (target_width, target_height), interpolation=interpolation
            )
        else:
            # OpenCV expects the target size in (width, height) order.
            target_width, target_height = self.size
            return resize_image(
                img, (target_width, target_height), interpolation=interpolation
            )

    def __repr__(self):
        if isinstance(self.size, int):
            return "Scale [Shortest Side ({:d})]".format(self.size)
        else:
            target_size = self.size
            return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])


class RandomColor(object):
    """Perturb color channels of a given image.

    This class will apply random color perturbation to an input image. An alpha
    value is first sampled uniformly from the range of (-r, r). 1 + alpha is
    further multiply to a color channel. The sampling is done independently for
    each channel. An efficient implementation can be achieved using a LuT.

    Args:
        color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
    """

    def __init__(self, color_range):
        self.color_range = color_range

    def __call__(self, img):
        # Work in floating point so multiplication cannot overflow uint8.
        jittered = img.astype(np.float32, copy=True)

        # Sample a different multiplier for every color channel.
        for channel in range(jittered.shape[2]):
            alpha = random.uniform(-self.color_range, self.color_range)
            jittered[:, :, channel] *= 1.0 + alpha

        # Images loaded by this project use the uint8 range [0, 255].
        jittered = np.clip(jittered, 0, 255)
        return jittered.astype(img.dtype)

    def __repr__(self):
        return "Random Color [Range {:.2f} - {:.2f}]".format(
            1 - self.color_range, 1 + self.color_range
        )


class RandomRotate(object):
    """Rotate the given numpy array (around the image center) by a random degree.

    This class will randomly rotate an image and further crop a local region with
    maximum area. A rotation angle is first sampled and then applied to the input.
    A region with maximum area and without any empty pixel is further determined
    and cropped.

    Args:
        degree_range (float): range of degree (-d ~ +d)
    """

    def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
        self.degree_range = degree_range
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]
        # sample rotation
        degree = random.uniform(-self.degree_range, self.degree_range)
        # ignore small rotations
        if np.abs(degree) <= 1.0:
            return img

        height, width = img.shape[:2]
        image_center = (width / 2.0, height / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, degree, 1.0)

        # Expand the canvas so the rotation itself does not clip the image.
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        rotated_width = int(np.ceil(width * abs_cos + height * abs_sin))
        rotated_height = int(np.ceil(width * abs_sin + height * abs_cos))
        rotation_matrix[0, 2] += rotated_width / 2.0 - image_center[0]
        rotation_matrix[1, 2] += rotated_height / 2.0 - image_center[1]

        rotated = cv2.warpAffine(
            img,
            rotation_matrix,
            (rotated_width, rotated_height),
            flags=interpolation,
        )

        # Compute the largest centered, axis-aligned rectangle that contains
        # only pixels from the original image (no empty rotation corners).
        angle = math.radians(degree)
        sin_angle = abs(math.sin(angle))
        cos_angle = abs(math.cos(angle))
        width_is_longer = width >= height
        long_side = max(width, height)
        short_side = min(width, height)

        if (
            short_side <= 2.0 * sin_angle * cos_angle * long_side
            or abs(sin_angle - cos_angle) < 1e-10
        ):
            half_short_side = 0.5 * short_side
            crop_width = (
                half_short_side / sin_angle
                if width_is_longer
                else half_short_side / cos_angle
            )
            crop_height = (
                half_short_side / cos_angle
                if width_is_longer
                else half_short_side / sin_angle
            )
        else:
            cos_double_angle = cos_angle * cos_angle - sin_angle * sin_angle
            crop_width = (
                width * cos_angle - height * sin_angle
            ) / cos_double_angle
            crop_height = (
                height * cos_angle - width * sin_angle
            ) / cos_double_angle

        crop_width = max(1, min(rotated_width, int(math.floor(crop_width))))
        crop_height = max(1, min(rotated_height, int(math.floor(crop_height))))
        left = (rotated_width - crop_width) // 2
        top = (rotated_height - crop_height) // 2

        return rotated[top : top + crop_height, left : left + crop_width]

    def __repr__(self):
        return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range
        )


#################################################################################
# Additional helper functions. No need to modify.
#################################################################################
class ToTensor(object):
    """Convert a ``numpy.ndarray`` image to tensor.
    Converts a numpy.ndarray (H x W x C) image in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # convert image to tensor
        assert (img.ndim > 1) and (img.ndim <= 3)
        if img.ndim == 2:
            img = img[:, :, None]
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        if img.ndim == 3:
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        # backward compatibility
        if isinstance(tensor_img, torch.ByteTensor):
            return tensor_img.float().div(255.0)
        else:
            return tensor_img


class Resize(object):
    """
    Resize an input image into a fixed resolution.
    """
    def __init__(self, size):
        assert (
            isinstance(size, int)
            or (isinstance(size, collections.Iterable) and len(size) == 2)
           )
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        img = resize_image(img, self.size, cv2.INTER_LINEAR)
        return img


class CenterCrop(object):
    """
    Crop the center patch of a given image (numpy array)

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
    """

    def __init__(
        self,
        size,
    ):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        if isinstance(self.size, int):
            th, tw = self.size, self.size
        else:
            th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        img = img[y1 : y1 + th, x1 : x1 + tw]
        return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return "Random Crop" + "[Size ({:d}, {:d})]".format(
            target_size[0],
            target_size[1],
        )


class SimpleDataset(data.Dataset):
    """
    A simple dataset using PyTorch dataloader (from our tutorial)
    """
    def __init__(self, root_folder, file_ext, transforms=None):
        # root folder, split
        self.root_folder = root_folder
        self.transforms = transforms
        self.file_ext = file_ext

        # list all files
        file_list = glob.glob(os.path.join(root_folder, '*.{:s}'.format(file_ext)))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load img and label (from file name)
        filename = self.file_list[index]
        img = load_image(filename)
        label = 1 if "cat" in os.path.basename(filename) else 0
        # apply data augmentation
        if self.transforms is not None:
            img  = self.transforms(img)
        return img, label


class Net(nn.Module):
    """
    A simple neural network modified from PyTorch tutorial
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
