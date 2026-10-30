import os
import random

import numpy as np

import torch
import torchvision

from .transforms import Compose, ConvertAnnotations, RandomHorizontalFlip, ToTensor


def trivial_batch_collator(batch):
    """
    A batch collator that allows us to bypass auto batching
    """
    return tuple(zip(*batch))


def worker_init_reset_seed(worker_id):
    """
    Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class VOCDetection(torchvision.datasets.CocoDetection):
    """
    A simple dataset wrapper to load VOC data
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        # eval_det reads this instead of guessing "<split>.json". COCO's
        # official files are named instances_train2017.json, so one path
        # pattern does not cover both datasets.
        self.ann_file = ann_file

    def get_cls_names(self):
        cls_names = (
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
        return cls_names

    @property
    def num_classes(self):
        return len(self.get_cls_names())

    def to_coco_category_id(self, label):
        # VOC json ids are already 1..20, in the same order as the head.
        # Inference emits that id directly (0-based index plus 1).
        return int(label)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class COCODetection(torchvision.datasets.CocoDetection):
    """COCO instances, with category ids packed into a dense range.

    A COCO category id is not "the class index plus one". The ids go up to
    90 and skip 12 numbers, because some categories were removed between
    the original release and the 80-class detection set. The classification
    head has one channel per class, so it needs indexes 0, 1, 2, ... with
    no holes. This class renames each json category id to 1..K before
    ConvertAnnotations subtracts 1. At evaluation, to_coco_category_id
    reverses that so the json written for pycocotools uses the original ids.
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.ann_file = ann_file
        categories = sorted(
            self.coco.loadCats(self.coco.getCatIds()), key=lambda cat: cat["id"]
        )
        self.cat_ids = [cat["id"] for cat in categories]
        self.cls_names = tuple(cat["name"] for cat in categories)
        # 1-based so the existing ConvertAnnotations "- 1" lands on 0..K-1.
        self._to_contiguous = {
            cat_id: index + 1 for index, cat_id in enumerate(self.cat_ids)
        }

    def get_cls_names(self):
        return self.cls_names

    @property
    def num_classes(self):
        return len(self.cat_ids)

    def to_coco_category_id(self, label):
        # `label` is the 1-based contiguous index produced by inference.
        return int(self.cat_ids[int(label) - 1])

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        annotations = []
        for obj in target:
            contiguous_id = self._to_contiguous.get(obj["category_id"])
            if contiguous_id is None:
                # Not one of the instance categories (should not happen for
                # instances_*.json). Dropping it is safer than training the
                # head on an id it has no channel for.
                continue
            obj = dict(obj)
            obj["category_id"] = contiguous_id
            annotations.append(obj)
        target = dict(image_id=image_id, annotations=annotations)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class DetectionSubset(torch.utils.data.Dataset):
    """First `max_images` entries of a seeded shuffle.

    COCO train2017 is about 118k images. Four epochs of that is a long
    laptop run, and the assignment explicitly allows a subset. The shuffle
    uses a fixed seed so every launch, including a resume, trains on the
    same images. The loader still shuffles this subset each epoch.

    Eval calls get_cls_names, to_coco_category_id, and ann_file on whatever
    object the loader holds, so those are forwarded even though training
    is the only place this wrapper is applied.
    """

    def __init__(self, dataset, max_images, seed=0):
        max_images = min(int(max_images), len(dataset))
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(dataset), generator=generator)
        self.indices = permutation[:max_images].tolist()
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def get_cls_names(self):
        return self.dataset.get_cls_names()

    def to_coco_category_id(self, label):
        return self.dataset.to_coco_category_id(label)

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def ann_file(self):
        return self.dataset.ann_file


def _resolve_ann_file(json_folder, split):
    """VOC stores trainval.json. COCO stores instances_train2017.json."""
    candidates = [
        os.path.join(json_folder, split + ".json"),
        os.path.join(json_folder, "instances_" + split + ".json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No annotation json for split %r in %s. Looked for %s."
        % (split, json_folder, ", ".join(os.path.basename(path) for path in candidates))
    )


def _resolve_img_folder(img_folder, split):
    """COCO keeps train and val images in separate directories.

    VOC keeps every JPEG in one folder, so the split subdirectory will
    not exist and the original folder is used.
    """
    split_dir = os.path.join(img_folder, split)
    if os.path.isdir(split_dir):
        return split_dir
    return img_folder


def build_dataset(name, split, img_folder, json_folder, max_images=None):
    """
    Create a detection dataset with the default train or eval transforms.
    New datasets can be linked here.
    """
    if name == "VOC2007":
        if split not in ("trainval", "test"):
            raise ValueError("VOC2007 split must be 'trainval' or 'test', got %r" % split)
        is_training = split == "trainval"
        dataset_cls = VOCDetection
    elif name == "COCO":
        if split not in ("train2017", "val2017"):
            raise ValueError("COCO split must be 'train2017' or 'val2017', got %r" % split)
        is_training = split == "train2017"
        dataset_cls = COCODetection
    else:
        raise ValueError("Unsupported dataset: %s" % name)

    if is_training:
        transforms = Compose([ConvertAnnotations(), RandomHorizontalFlip(), ToTensor()])
    else:
        transforms = Compose([ConvertAnnotations(), ToTensor()])

    ann_file = _resolve_ann_file(json_folder, split)
    image_dir = _resolve_img_folder(img_folder, split)
    dataset = dataset_cls(image_dir, ann_file, transforms)
    # Subset the training split only. Val and test stay complete so the
    # mAP is on the split the config names, not on a random slice of it.
    if is_training and max_images:
        full_count = len(dataset)
        dataset = DetectionSubset(dataset, max_images, seed=0)
        print(
            "Training on a fixed subset of %d / %d %s images (seed 0)."
            % (len(dataset), full_count, name)
        )
    return dataset


def build_dataloader(dataset, is_training, batch_size, num_workers):
    """
    Create a dataloder for VOC dataset
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        persistent_workers=(num_workers > 0),
    )
    return loader
