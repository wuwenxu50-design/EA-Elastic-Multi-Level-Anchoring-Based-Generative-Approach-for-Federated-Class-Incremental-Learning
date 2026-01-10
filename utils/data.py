import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.tiny_imagenet import TinyImageNet
import os


#data_dir = os.path.join(os.environ['HOME'],"../datasets")
data_dir = os.path.join(os.environ['HOME'],"datasets")
# data_dir = "../../data"
def make_imbalanced_cifar(data, targets, imb_type="exp", imb_factor=0.01, seed=0, num_classes=100):
    """
    Create CIFAR-100-LT from CIFAR-100 train set.
    - data: numpy array (N, H, W, C)
    - targets: numpy array (N,)
    - imb_factor: min/max ratio (e.g., 0.1 => IF~10, 0.02 => IF~50, 0.01 => IF~100)
    """
    rng = np.random.RandomState(seed)
    targets = np.array(targets)

    img_max = int(len(data) / num_classes)  # CIFAR-100 train: 500 per class

    if imb_type == "exp":
        img_num_per_cls = [
            int(round(img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))))
            for cls_idx in range(num_classes)
        ]
    elif imb_type == "step":
        img_num_per_cls = [img_max] * (num_classes // 2) + \
                          [int(round(img_max * imb_factor))] * (num_classes - num_classes // 2)
    else:
        raise ValueError(f"Unknown imb_type: {imb_type}")

    new_data, new_targets = [], []
    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        sel = cls_idx[:img_num_per_cls[cls]]
        new_data.append(data[sel])
        new_targets.append(targets[sel])

    new_data = np.concatenate(new_data, axis=0)
    new_targets = np.concatenate(new_targets, axis=0)

    perm = rng.permutation(len(new_targets))
    return new_data[perm], new_targets[perm], img_num_per_cls


class iData(object):
    def __init__(self, args):
        self.args = args
        self.train_trsf = []
        self.test_trsf = []
        self.common_trsf = []
        self.class_order = None

    def get_train_trsf(self):
        return self.train_trsf

    def get_test_trsf(self):
        return self.test_trsf

    def get_common_trsf(self):
        return self.common_trsf

    def get_class_order(self):
        return self.class_order


class iCIFAR10(iData):
    def __init__(self, args):
        super(iCIFAR10, self).__init__(args)
        self.use_path = False
        self.train_trsf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
        ]
        self.test_trsf = []
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ]

        self.class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):

    def __init__(self, args):
        super(iCIFAR100, self).__init__(args)
        self.use_path = False
        self.train_trsf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor()
        ]
        self.test_trsf = [transforms.ToTensor()]
        self.common_trsf = [
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            ),
        ]
        self.class_order = [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80, 2, 47, 87, 78, 66, 19, 6, 24, 10, 59, 30, 22, 29, 83, 37, 93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 84, 70, 4, 20, 15, 21, 31, 76, 57, 67, 73, 50, 69, 25, 98, 46, 96, 0, 72, 35, 58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9, 82, 39, 18, 77, 68, 32, 79, 12, 85, 36, 17, 64, 27, 74, 45, 61, 38, 51, 62, 65, 33, 5, 53, 97, 49]
        # self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        imb_type = self.args.get("imb_type", "none")
        imb_factor = float(self.args.get("imb_factor", 1.0))
        imb_seed = int(self.args.get("imb_seed", 0))

        if imb_type != "none" and imb_factor < 1.0:
            self.train_data, self.train_targets, img_num_per_cls = make_imbalanced_cifar(
                self.train_data, self.train_targets,
                imb_type=imb_type, imb_factor=imb_factor, seed=imb_seed, num_classes=100
            )
            print(f"[CIFAR-100-LT] imb_type={imb_type}, imb_factor={imb_factor}, seed={imb_seed}")
            print(f"[CIFAR-100-LT] min/max per-class counts: {min(img_num_per_cls)}/{max(img_num_per_cls)}")


class iImageNet1000(iData):
    def __init__(self, args):
        super(iImageNet1000, self).__init__(args)
        self.use_path = True
        self.train_trsf = [
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(1000).tolist()

    def download_data(self):
        data_root = os.path.join(data_dir, 'imagenet')
        train_root = os.path.join(data_root, 'train')
        val_root = os.path.join(data_root, 'val')

        train_dset = datasets.ImageFolder(train_root)
        test_dset = datasets.ImageFolder(val_root)

        # train_dset = datasets.ImageNet(data_root, split='train', transform=self.train_trsf)
        # test_dset = datasets.ImageNet(data_root, split='val', transform=self.test_trsf)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):  # 1300*100 = 13w, 5 tasks, each task 20*1300=2.6w
    def __init__(self, args):
        super(iImageNet100, self).__init__(args)
        self.use_path = True
        self.train_trsf = [
            # transforms.RandomResizedCrop(224),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.CenterCrop(128),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = "{}/imagenet100/train/".format(data_dir)
        test_dir = "{}/imagenet100/val/".format(data_dir)

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


# transforms.RandomResizedCrop(64, scale=(0.4,1.0))
class TinyImageNet200(iData):   # 200*500=10w, 5 tasks, each task=40*500=2w
    def __init__(self, args):
        super(TinyImageNet200, self).__init__(args)
        self.use_path = True
        self.train_trsf = [
            # transforms.RandomResizedCrop(64),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.CenterCrop(64),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
        self.class_order = np.arange(200).tolist()


    def download_data(self):
        train_dst = TinyImageNet(data_dir, split='train', transform=self.train_trsf, download=True)
        val_dst = TinyImageNet(data_dir, split='val', transform=self.test_trsf, download=True)

        self.train_data, self.train_targets = split_images_labels(train_dst.data)
        self.test_data, self.test_targets = split_images_labels(val_dst.data)
class iStanfordCars196(iData):
    def __init__(self, args):
        super(iStanfordCars196, self).__init__(args)
        self.use_path = True
        self.train_trsf = [
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        self.class_order = np.arange(196).tolist()

    def download_data(self):
        data_root = os.path.join(data_dir, 'stanford_cars')
        train_root = os.path.join(data_root, 'train')
        test_root  = os.path.join(data_root, 'test')

        train_dset = datasets.ImageFolder(train_root)
        test_dset  = datasets.ImageFolder(test_root)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data,  self.test_targets  = split_images_labels(test_dset.imgs)



