import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation
import time, os, math
import torch.nn.init as init
from PIL import Image
import pickle
from methods.generator import NLGenerator, NLGenerator_IN
import shutil
import glob
bn_mmt = 0.9
tau = 2
T = 20.0
student_train_step = 50

def get_norm_and_transform(dataset):
    if dataset == "cifar100":
        data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
    elif dataset == "tiny_imagenet":
        data_normalize = dict(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
    elif dataset == "imagenet":
        data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(**dict(data_normalize)),
        ])
    elif dataset == "stanford_cars":
        data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(**dict(data_normalize)),
        ])
    normalizer = Normalizer(**dict(data_normalize))
    return train_transform, normalizer


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


# normalizer = Normalizer(**dict(data_normalize))


def _collect_all_images(nums, root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            if nums != None:
                files.sort()
                files = files[:nums]
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        # print("1: %s" % str(np.array(img).shape))
        if self.transform:
            img = self.transform(img)
        # print("2: %s" % str(img.shape))
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
        self.root, len(self), self.transform)


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67)  # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)  # , alpha=0.67


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '-%d.png' % (idx))


class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data)

    def remove(self):
        self.hook.remove()


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)


def custom_cross_entropy(preds, target):
    return torch.mean(torch.sum(-target * preds.log_softmax(dim=-1), dim=-1))


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


class NAYER():
    def __init__(self, teacher, student, generator, num_classes, img_size, iterations=100, lr_g=0.1, label_emb=None,
                 synthesis_batch_size=128, adv=0.0, bn=1, oh=1, r=1e-1,radius_hat=None,ltc=0.2, save_dir='run/fast', transform=None,
                 normalizer=None, device="gpu:0", warmup=10, bn_mmt=0, args=None,anchor2_pool=None, K2=6):
        super(NAYER, self).__init__()
        self.teacher = teacher
        self.student = student
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.args = args
        self.label_emb = label_emb
        self.ltc = ltc
        self.r = r
        self.radius_hat = radius_hat


        self.K2 = int(K2)

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.normalizer = normalizer

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.device = device

        self.generator = generator.to(self.device).train()

        self.anchor2_pool = None
        if anchor2_pool is not None:
            self.anchor2_pool = anchor2_pool.to(self.device, dtype=self.label_emb.dtype)
        self.ep = 0
        self.ep_start = self.args['warmup'] + 1 # need more time to the student train well.
        self.prev_z = None

        if self.args["dataset"] == "imagenet":
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
            ])
        else:
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ])

        self.mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=True, device=self.device)
        self.std = torch.tensor([0.2, 0.2, 0.2], requires_grad=True, device=self.device)

        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

    def synthesize(self, _cur_task=0):
        self._cur_task = _cur_task
        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        best_oh = 1e6

        best_inputs = None
        self.generator.re_init_le()


        K2 = int(self.K2)

        base_bs = max(1, int(self.synthesis_batch_size) // K2)
        targets0, ys0 = self.generate_ys(cr=0.0, batch_size=base_bs)
        targets0 = targets0.to(self.device)
        ys0 = ys0.to(self.device)
        cond0 = self.anchor2_pool[targets0]  # [B0, K2, D]
        cond = cond0.reshape(-1, cond0.shape[-1])  # [B0*K2, D]

        targets = targets0.repeat_interleave(K2)  # [B0*K2]
        ys = ys0.repeat_interleave(K2, dim=0)  # [B0*K2, C]

        optimizer = torch.optim.Adam([
            {'params': self.generator.parameters()},
            {'params': [self.mean], 'lr': 0.01},
            {'params': [self.std], 'lr': 0.01}
        ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):
            inputs = self.generator(targets=targets, cond=cond)
            inputs_aug = self.aug(inputs)
            inputs_aug = (inputs_aug - self.mean[None, :, None, None]) / (self.std[None, :, None, None])
            output_list = self.teacher(inputs_aug)
            t_out = output_list["logits"]
            feature = output_list["att"]

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = custom_cross_entropy(t_out, ys.detach())

            if self.adv > 0 and (self.ep - 1 > self.ep_start):
                s_out = self.student(inputs_aug)["logits"]
                mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()
            else:
                loss_adv = loss_oh.new_zeros(1)
            target_f = self.label_emb[targets]
            diff = feature - target_f.detach()
            d = (diff * diff).mean(dim=1)  # [B]

            if self.radius_hat is not None:
                r_y = self.radius_hat[targets]  # [B]
            else:
                r_y = torch.full_like(d, float(self.r))

            loss_f = F.relu(d - r_y).mean()
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.ltc * loss_f

            if loss_oh.item() < best_oh:
                best_oh = loss_oh

            print("%s - bn %s - bn %s - oh %s - adv %s -fr %s - %s - %s" % (
                it,
                float((loss_bn * self.bn).data.cpu().detach().numpy()),
                float(loss_bn.data.cpu().detach().numpy()),
                float((loss_oh).data.cpu().detach().numpy()),
                float((loss_adv).data.cpu().detach().numpy()),
                float(loss_f.data.cpu().detach().numpy()),
                str(self.mean.detach().cpu()[0]),
                str(self.std.detach().cpu()[0])))

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        if self.args['warmup'] <= self.ep:
            self.data_pool.add(best_inputs)
            self.student_train(self.student, self.teacher)

    def student_train(self, student, teacher):
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data()
        data_iter = DataIter(loader)
        criterion = KLDiv(T=T)

        prog_bar = tqdm(range(student_train_step))
        for _, com in enumerate(prog_bar):
            images = data_iter.next()
            images = images.to(self.device)
            with torch.no_grad():
                t_out_list = teacher(images)
                t_out = t_out_list["logits"]
                t_f = t_out_list["att"]
            s_out_list = student(images.detach())
            s_out = s_out_list["logits"]
            s_f = s_out_list["att"]
            loss_s = criterion(s_out, t_out.detach())
            s_loss_f = torch.nn.functional.mse_loss(s_f, t_f.detach())

            loss = loss_s + self.ltc*s_loss_f

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def get_all_syn_data(self):
        syn_dataset = UnlabeledImageDataset(self.save_dir, transform=self.transform, nums=self.args['nums'])
        loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=self.args["local_bs"], shuffle=True, persistent_workers=True,
            num_workers=self.args["num_worker"])
        return loader

    def generate_ys(self, cr=0.0, batch_size=None):
        if batch_size is None:
            batch_size = self.synthesis_batch_size
        s = batch_size // self.num_classes
        v = batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        for i in range(s):
            tmp_label = torch.tensor(range(0, self.num_classes))
            target = torch.cat((tmp_label, target))
        ys = torch.zeros(batch_size, self.num_classes)
        ys.fill_(cr / (self.num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))
        return target, ys

    def get_student(self):
        return self.student

    def get_teacher(self):
        return self.teacher


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)

        return data


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).cuda()
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)
    logits = torch.gather(logits, 1, nt_positions)

    return logits


class EA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.class_order = torch.tensor(args["class_order"], device=args["gpu"])
        le_name = os.path.join("label_embedding", f"{args['dataset']}_le.pickle")
        with open(le_name, "rb") as label_file:
            label_emb = pickle.load(label_file)
            order = torch.as_tensor(args["class_order"], dtype=torch.long)
            if not isinstance(label_emb, torch.Tensor):
                label_emb = torch.as_tensor(label_emb)
            label_emb = label_emb[order].float().detach()
            self.label_emb = label_emb.to(args["gpu"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if isinstance(label_emb, torch.Tensor):
                label_emb = label_emb.to(device)
            if isinstance(self.class_order, torch.Tensor):
                idx = self.class_order.to(device=device, dtype=torch.long)
            else:
                idx = torch.tensor(self.class_order, device=device, dtype=torch.long)

            label_emb = label_emb.index_select(0, idx)
            label_emb = label_emb[self.class_order]
            self.label_emb = label_emb.to(args['gpu']).float().detach()
        self.r = args['r']
        self.ltc = args['ltc']
        self.transform, self.normalizer = get_norm_and_transform(self.args["dataset"])
        # ===== Multi-level anchors =====
        self.K2 = int(self.args.get("K2", 3))
        self.kmeans_iter = int(self.args.get("kmeans_iter", 10))
        self.kmeans_max_per_class = int(self.args.get("kmeans_max_per_class", 256))
        self.anchor2_pool = None  # shape: [C, K2, D]

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        test_acc = self._compute_accuracy(self._old_network, self.test_loader)
        print("After Test Acc: %s" % test_acc)

    def _init_anchor2_pool_if_needed(self):
        device = self.args["gpu"]
        C = int(self._total_classes)
        K = int(self.K2)
        D = int(self.label_emb.shape[1])

        if (self.anchor2_pool is None) or (self.anchor2_pool.numel() == 0):
            self.anchor2_pool = torch.zeros((C, K, D), device=device, dtype=self.label_emb.dtype)
            for y in range(C):
                self.anchor2_pool[y] = self.label_emb[y].unsqueeze(0).repeat(K, 1)
            return

        oldC = int(self.anchor2_pool.shape[0])
        if C > oldC:
            new_pool = torch.zeros((C, K, D), device=device, dtype=self.anchor2_pool.dtype)
            new_pool[:oldC] = self.anchor2_pool
            for y in range(oldC, C):
                new_pool[y] = self.label_emb[y].unsqueeze(0).repeat(K, 1)
            self.anchor2_pool = new_pool

    @torch.no_grad()
    def _kmeans_torch(self, X: torch.Tensor, K: int, iters: int = 10) -> torch.Tensor:
        """
        X: [N, D]  -> return centers [K, D]
        """
        assert X.dim() == 2
        N, D = X.shape
        if N == 0:
            raise ValueError("_kmeans_torch got empty X")
        if N < K:
            reps = (K + N - 1) // N
            X_pad = X.repeat((reps, 1))[:K]
            return X_pad
        idx = torch.randperm(N, device=X.device)[:K]
        centers = X[idx].clone()

        for _ in range(iters):
            # dist: [N, K]
            dist = torch.cdist(X, centers, p=2)
            assign = dist.argmin(dim=1)  # [N]

            new_centers = []
            for k in range(K):
                mask = (assign == k)
                if mask.any():
                    new_centers.append(X[mask].mean(dim=0))
                else:
                    new_centers.append(centers[k])
            centers = torch.stack(new_centers, dim=0)

        return centers

    @torch.no_grad()
    def _collect_hiconf_features_by_class(self, model, loader):
        model.eval()
        device = self.args["gpu"]
        use_sr = int(self.args.get("use_soft_radius", 0)) == 1 and hasattr(model,
                                                                           "radius_raw") and model.radius_raw is not None
        feats = {}  # y -> list[tensor]

        for _, images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            f = out["att"]  # [B, D]

            e1 = self.label_emb[labels]  # [B, D]
            d = ((f - e1) ** 2).mean(dim=1)  # [B]

            if use_sr:
                r_hat = torch.nn.functional.softplus(model.radius_raw)  # [C]
                r_y = r_hat[labels]  # [B]
            else:
                r_y = torch.full_like(d, float(self.r))
            mask = (d <= r_y)
            if mask.any():
                f = f[mask]
                y = labels[mask]
            else:
                continue


            for cls in y.unique():
                cls = int(cls.item())
                cls_mask = (y == cls)
                feats.setdefault(cls, []).append(f[cls_mask].detach().cpu())


        out_dict = {}
        for cls, chunks in feats.items():
            X = torch.cat(chunks, dim=0)  # [N, D] on CPU
            if X.shape[0] > self.kmeans_max_per_class:
                idx = torch.randperm(X.shape[0])[:self.kmeans_max_per_class]
                X = X[idx]
            out_dict[cls] = X

        return out_dict

    def remove_syn_imgs(self):
        folder_path = os.path.join(self.save_dir, "task_{}".format(self._cur_task-1))
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (folder_path, e))

    def data_generation(self):
        if self.args["dataset"] == "cifar100":
            img_size = 32
            img_shape = (3, 32, 32)
            generator = NLGenerator(ngf=64, img_size=img_size, nc=3, nl=10,
                                    label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                    sbz=self.args['synthesis_batch_size'],film_enabled=True,film_hidden=int(self.args.get("film_hidden", self.label_emb.shape[1])))
        elif self.args["dataset"] == "tiny_imagenet":
            img_size = 64
            img_shape = (3, 64, 64)
            generator = NLGenerator(ngf=64, img_size=img_size, nc=3, nl=10,
                                    label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                    sbz=self.args['synthesis_batch_size'],film_enabled=True,film_hidden=int(self.args.get("film_hidden", self.label_emb.shape[1])))
        elif self.args["dataset"] == "imagenet":
            img_size = 224
            img_shape = (3, 224, 224)
            generator = NLGenerator_IN(ngf=64, img_size=img_size, nc=3, nl=10,
                                      label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                      sbz=self.args['synthesis_batch_size'],film_enabled=True,film_hidden=int(self.args.get("film_hidden", self.label_emb.shape[1])))
        elif self.args["dataset"] == "stanford_cars":
            img_size = 224
            img_shape = (3, 224, 224)
            generator = NLGenerator_IN(
                ngf=64, img_size=img_size, nc=3, nl=10,
                label_emb=self.label_emb, le_emb_size=self.args['nz'],
                sbz=self.args['synthesis_batch_size'],
                film_enabled=True,
                film_hidden=int(self.args.get("film_hidden", self.label_emb.shape[1])),
            )
        student = copy.deepcopy(self._network)
        student.apply(weight_init)
        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        #print("[DBG] save_dir:", self.save_dir)
        #print("[DBG] tmp_dir:", tmp_dir)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        radius_hat = None
        if int(self.args.get("use_soft_radius", 0)) == 1 and hasattr(self._network,
                                                                     "radius_raw") and self._network.radius_raw is not None:
            radius_hat = F.softplus(self._network.radius_raw).detach()
        self._init_anchor2_pool_if_needed()
        synthesizer = NAYER(copy.deepcopy(self._network), student, generator, num_classes=self._total_classes,
                            img_size=img_shape, save_dir=tmp_dir, transform=self.transform, normalizer=self.normalizer,
                            synthesis_batch_size=self.args['synthesis_batch_size'], iterations=self.args['g_steps'],
                            warmup=self.args['warmup'], lr_g=self.args['lr_g'], adv=self.args['adv'], bn=self.args['bn'],
                            oh=self.args['oh'], ltc=self.ltc, r=self.r,radius_hat=radius_hat,device=self.args["gpu"], bn_mmt=bn_mmt,
                            args=self.args, label_emb=self.label_emb,anchor2_pool=self.anchor2_pool.detach(),K2=self.K2)

        for it in range(self.args['syn_round'] + self.args['warmup']):
            synthesizer.synthesize(self._cur_task)  # generate synthetic data
            cnt = len(glob.glob(os.path.join(tmp_dir, "**", "*"), recursive=True))
            print("[DBG] files under tmp_dir:", cnt, "dir:", tmp_dir)
            if it > self.args['warmup']:
                ms = synthesizer.get_student()
                test_accs = self._compute_accuracy(ms, self.test_loader)
                mt = synthesizer.get_teacher()
                test_acct = self._compute_accuracy(mt, self.test_loader)
                print("Student Test Acc: %s - %s" % (test_accs, test_acct))
                wandb.log({'Task_{}, Student Accuracy'.format(self._cur_task): test_accs})

        print("For task {}, data generation completed! ".format(self._cur_task))

    def get_syn_data_loader(self):
        if self.args["dataset"] == "cifar100":
            dataset_size = 50000
        elif self.args["dataset"] == "tiny_imagenet":
            dataset_size = 100000
        elif self.args["dataset"] == "imagenet100":
            dataset_size = 130000
        elif self.args["dataset"] == "imagenet":
            dataset_size = 1000000
        iters = math.ceil(dataset_size / (self.args["num_users"] * self.args["tasks"] * self.args["local_bs"]))
        # syn_bs = int(self.nums / iters)
        syn_bs = self.args["syn_bs"]*self.args["local_bs"]
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task - 1))
        print("iters{}, syn_bs:{}, data_dir: {}".format(iters, syn_bs, data_dir))
        # print(syn_bs)
        syn_dataset = UnlabeledImageDataset(data_dir, transform=self.transform, nums=self.nums)
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=syn_bs, shuffle=True, persistent_workers=True,
            num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"])
        return syn_data_loader

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        # self.args["kd"] = self.args["kd"] * (2 - self._known_classes / self._total_classes)

        self._network.update_fc(self._total_classes)
        self._init_anchor2_pool_if_needed()
        self._network.cuda()
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(  # * get the data for one task
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"], persistent_workers=True
        )
        print(self.test_loader)
        if self._cur_task > 0:
            old_dataset = data_manager.get_dataset(
                np.arange(0, self._known_classes), source="test", mode="test"
            )
            self.old_loader = DataLoader(
                old_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"], persistent_workers=True
            )

            new_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="test", mode="test"
            )
            self.new_loader = DataLoader(
                new_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"], persistent_workers=True
            )

        setup_seed(self.seed)
        if self._cur_task == 0 and (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        if self._cur_task != 0:
            # get syn_data for old tasks
            self.syn_data_loader = self.get_syn_data_loader()

        # * for all tasks
        if self.args['type'] != -1 and self.args['type'] - 1 > self._cur_task:
            return 0
        elif self.args['type'] != -1 and self.args['type'] - 1 == self._cur_task:
            local_weights = torch.load('store/global_{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                                       .format(self.args["dataset"], self.args["net"], self.args["num_users"],
                                               self.args["beta"], self.args["method"], self._cur_task, self.ltc,
                                               self.args['exp_name']))
            self._network.load_state_dict(local_weights)
            self._network.cuda()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            if self._cur_task > 0:
                test_old_acc = self._compute_accuracy(copy.deepcopy(self._network), self.old_loader)
                test_new_acc = self._compute_accuracy(copy.deepcopy(self._network), self.new_loader)
                print("Task {}, Test_accy {:.2f} O {} N {}".format(self._cur_task, test_acc, test_old_acc,
                                                                   test_new_acc))
            print("Task {} =>  Test_accy {:.2f}".format(self._cur_task, test_acc, ))
            if self.wandb == 1:
                wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})
        else:
            self._fl_train(train_dataset, self.test_loader)
        if self._cur_task + 1 != self.tasks:
            if self.args['type'] == -1 or (self.args['type'] > -1 and self._cur_task >= self.args['type']) or self.args['syn']:
                torch.save(copy.deepcopy(self._network.state_dict()), 'store/global_{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                           .format(self.args["dataset"], self.args["net"], self.args["num_users"],
                                   self.args["beta"], self.args["method"], self._cur_task, self.ltc, self.args['exp_name']))
                self.data_generation()

    @torch.no_grad()
    def _server_update_anchor2_pool(self, client_anchor2_updates):
        if len(client_anchor2_updates) == 0:
            return

        self._init_anchor2_pool_if_needed()
        K = int(self.K2)

        per_class = {}  # y -> list[tensor[K2, D]]
        for d in client_anchor2_updates:
            for y, a in d.items():
                per_class.setdefault(int(y), []).append(a)

        for y, arr in per_class.items():
            X = torch.cat(arr, dim=0)  # [M*K2, D]
            if X.shape[0] >= K:
                centers = self._kmeans_torch(X, K, iters=self.kmeans_iter)
            else:
                centers = self._kmeans_torch(X, K, iters=self.kmeans_iter)
            self.anchor2_pool[y] = centers

    def _fl_train(self, train_dataset, test_loader):
        self._network.cuda()
        self.best_model = None  # Best model using the lowest training loss
        self.lowest_loss = np.inf
        user_groups, _ = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        prog_bar = tqdm(range(self.args["com_round"]))
        optimizer = torch.optim.SGD(self._network.parameters(), lr=self.args['local_lr'], momentum=0.9, weight_decay=self.args['weight_decay'])
        if self.args["dataset"] == "tiny_imagenet" or self.args["dataset"] == "imagenet":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args["com_round"], eta_min=1e-3)
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            loss_weight = []
            client_anchor2_updates = []
            for idx in idxs_users:
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                                                batch_size=self.args["local_bs"], shuffle=True, num_workers=self.args["num_worker"],
                                                pin_memory=True, multiprocessing_context=self.args["mulc"], persistent_workers=True)
                if self._cur_task == 0:
                    w, total_loss = self._local_update(copy.deepcopy(self._network), local_train_loader, scheduler.get_last_lr()[0])
                else:
                    w, total_syn, total_local, total_loss = self._local_finetune(self._old_network,
                                                                                 copy.deepcopy(self._network),
                                                                                 local_train_loader, self._cur_task,
                                                                                 idx, scheduler.get_last_lr()[0])
                    if com == 0 and self._cur_task == 1:
                        print("\t \t client {}, local dataset size:{},  syntheic data size:{}".format(idx, total_local,
                                                                                                      total_syn))
                
                tmp_model = copy.deepcopy(self._network).cuda()
                tmp_model.load_state_dict(w)
                tmp_model.eval()

                feat_dict = self._collect_hiconf_features_by_class(tmp_model, local_train_loader)

                anchor2_dict = {}
                for y, X_cpu in feat_dict.items():
                    
                    if y < self._known_classes or y >= self._total_classes:
                        continue

                    X = X_cpu.to(self.args["gpu"], dtype=self.label_emb.dtype)
                    centers = self._kmeans_torch(X, self.K2, iters=self.kmeans_iter)  # [K2, D]
                    anchor2_dict[int(y)] = centers.detach()

                if len(anchor2_dict) > 0:
                    client_anchor2_updates.append(anchor2_dict)

                del tmp_model
                torch.cuda.empty_cache()
                local_weights.append(copy.deepcopy(w))
                loss_weight.append(total_loss)
                del local_train_loader, w
                torch.cuda.empty_cache()
            scheduler.step()
            sum_loss = sum(loss_weight)  # total loss of previous model
            if sum_loss < self.lowest_loss:
                self.lowest_loss = sum_loss
                self.best_model = copy.deepcopy(self._network.state_dict())

            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            self._server_update_anchor2_pool(client_anchor2_updates)
            if com % 1 == 0 and com < self.args["com_round"]:
                log_root = os.path.join(self.save_dir, "batch_acc_logs", f"K{self.K2}", f"task_{self._cur_task}")
                log_path_test = None
                if int(self.K2) == 3:
                    log_path_test = os.path.join(log_root, f"com_{com+1:03d}_test.csv")
                test_acc = self._compute_accuracy(self._network, test_loader, log_path=log_path_test)
                if self._cur_task > 0:
                    test_old_acc = self._compute_accuracy(copy.deepcopy(self._network), self.old_loader)
                    test_new_acc = self._compute_accuracy(copy.deepcopy(self._network), self.new_loader)
                    print("Task {}, Test_accy {:.2f} O {} N {}".format(self._cur_task, test_acc, test_old_acc,
                                                                       test_new_acc))
                info = ("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc, ))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})
        self._network.load_state_dict(self.best_model)  # Best model using the lowest training loss
        del self.best_model
        torch.cuda.empty_cache()

    def _local_update(self, model, train_data_loader, lr):
        print(lr)
        model.train()
        total_loss = 0
        total_ce_loss = 0
        total_f_loss = 0
        use_sr = int(self.args.get("use_soft_radius", 0)) == 1 and hasattr(model,
                                                                           "radius_raw") and model.radius_raw is not None

        main_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if use_sr and name == "radius_raw":
                continue
            main_params.append(p)

        wd = float(self.args.get("weight_decay", 0.0))
        optimizer_main = torch.optim.SGD(
            main_params, lr=lr, momentum=0.9, weight_decay=wd
        )

        optimizer_r = None
        if use_sr:
            lr_r = float(self.args.get("lr_r", lr))
            optimizer_r = torch.optim.SGD([model.radius_raw], lr=lr_r, momentum=0.0, weight_decay=0.0)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                # print(images.shape)
                output_list = model(images)
                if batch_idx == 0 and iter == 0 and use_sr:
                    r_hat_dbg = F.softplus(model.radius_raw).detach()
                    print("[DBG] radius_hat: min {:.6f} mean {:.6f} max {:.6f}".format(
                        r_hat_dbg.min().item(), r_hat_dbg.mean().item(), r_hat_dbg.max().item()
                    ))
                output = output_list["logits"]
                feature = output_list["att"]
                target_f = self.label_emb[labels]  # [B, D]
                diff = feature - target_f.detach()
                d = (diff * diff).mean(dim=1)  # [B]
                if use_sr:
                    r_hat = F.softplus(model.radius_raw)  # [C]
                    r_y = r_hat[labels]  # [B]
                else:
                    r_y = torch.full_like(d, float(self.r))
                loss_ce = F.cross_entropy(output, labels)
                loss_f = F.relu(d - r_y.detach()).mean()
                loss_main = loss_ce + self.ltc * loss_f
                optimizer_main.zero_grad()
                loss_main.backward()
                optimizer_main.step()
                if use_sr and optimizer_r is not None:
                    d_det = d.detach()
                    r_hat = F.softplus(model.radius_raw)
                    r_y = r_hat[labels]
                    l_bound = torch.where(d_det > r_y, d_det - r_y, r_y - d_det)
                    loss_r = l_bound.mean()
                    r_reg = float(self.args.get("r_reg", 0.0))
                    loss_r = loss_r + r_reg * r_y.mean()
                    optimizer_r.zero_grad()
                    loss_r.backward()
                    optimizer_r.step()
                if iter == 0:
                    total_loss += loss_main.detach()
                    total_ce_loss += loss_ce.detach()
                    total_f_loss += loss_f.detach()
        print("---task {} =>  CE: {} F: {}, CE: {}, F: {} T: {}".format(self._cur_task, loss_ce.detach(), loss_f.detach(), total_ce_loss, total_f_loss, total_loss))

        return model.state_dict(), total_loss

    def _local_finetune(self, teacher, model, train_data_loader, task_id, client_id, lr):
        alpha = np.log2(self._total_classes / 2 + 1)
        beta = np.sqrt(self._known_classes / self._total_classes)
        cur = self.args["cur"] * (1 + 1 / alpha) / beta
        pre = self.args["pre"] * alpha * beta

        # global print_flag
        model.train()
        teacher.eval()
        total_loss = 0
        use_sr = int(self.args.get("use_soft_radius", 0)) == 1 and hasattr(model,
                                                                           "radius_raw") and model.radius_raw is not None
        main_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if use_sr and name == "radius_raw":
                continue
            main_params.append(p)

        wd = float(self.args.get("weight_decay", 0.0))
        optimizer_main = torch.optim.SGD(
            main_params, lr=lr, momentum=0.9, weight_decay=wd
        )
        optimizer_r = None
        if use_sr:
            lr_r = float(self.args.get("lr_r", lr))
            optimizer_r = torch.optim.SGD([model.radius_raw], lr=lr_r, momentum=0.0, weight_decay=0.0)
        syn_data_iter = DataIter(self.syn_data_loader)
        for it in range(self.args["local_ep"]):
            # iter_loader = enumerate(zip((train_data_loader), (self.syn_data_loader)))
            iter_loader = enumerate(train_data_loader)
            total_local = 0.0
            total_syn = 0.0
            total_ce_loss = 0
            total_f_loss = 0
            total_kd_loss = 0
            total_sf_loss = 0
            total_loss_ep0 = 0.0

            # for batch_idx, ((_, images, labels), syn_input) in iter_loader:
            for batch_idx, (_, images, labels) in iter_loader:
                images, labels = images.cuda(), labels.cuda()
                syn_batch = syn_data_iter.next()
                syn_input = syn_batch[0] if isinstance(syn_batch, (tuple, list)) else syn_batch
                syn_input = syn_input.cuda()
                # print("%s - %s" % (images.shape, syn_input.shape))
                fake_targets = labels - self._known_classes
                c_output_list = model(images)
                c_output = c_output_list["logits"]
                c_feature = c_output_list["att"]
                c_target_f = self.label_emb[labels]
                c_target_f = self.label_emb[labels]
                diff_c = c_feature - c_target_f.detach()
                d_c = (diff_c * diff_c).mean(dim=1)  # [B]

                if use_sr:
                    r_hat = F.softplus(model.radius_raw)  # [C]
                    r_y = r_hat[labels]  # [B]
                else:
                    r_y = torch.full_like(d_c, float(self.r))

                c_loss_f = F.relu(d_c - r_y.detach()).mean()  
                loss_ce_cur = F.cross_entropy(c_output[:, self._known_classes:], fake_targets)
                s_out_list = model(syn_input.detach())
                s_out = s_out_list["logits"]
                s_f = s_out_list["att"]
                with torch.no_grad():
                    t_out_list = teacher(syn_input.detach())
                    t_out = t_out_list["logits"]
                    t_f = t_out_list["att"]
                    total_syn += syn_input.shape[0]
                    total_local += images.shape[0]
                loss_kd = _KD_loss(s_out[:, : self._known_classes], t_out.detach(), tau)
                s_loss_f = torch.nn.functional.mse_loss(s_f, t_f.detach())
                loss_main = cur * (loss_ce_cur + self.ltc * c_loss_f) + pre * (loss_kd + s_loss_f)
                optimizer_main.zero_grad()
                loss_main.backward()
                optimizer_main.step()
                if use_sr and optimizer_r is not None:
                    d_det = d_c.detach()

                    r_hat = F.softplus(model.radius_raw)
                    r_y2 = r_hat[labels]

                    l_bound = torch.where(d_det > r_y2, d_det - r_y2, r_y2 - d_det)
                    loss_r = l_bound.mean()

                    r_reg = float(self.args.get("r_reg", 0.0))
                    loss_r = loss_r + r_reg * r_y2.mean()

                    optimizer_r.zero_grad()
                    loss_r.backward()
                    optimizer_r.step()
                if it == 0:
                    total_loss += loss_main.detach()
                    total_ce_loss += loss_ce_cur.detach()
                    total_f_loss += c_loss_f.detach()
                    total_kd_loss += loss_kd.detach()
                    total_sf_loss += s_loss_f.detach()
            if it == 0:
                total_loss += total_loss_ep0
                    
            print("---task {}, ep {}/{} =>  CE: {} F: {} CE: {}, F: {}. TKL: {}, TF: {}, T: {}".format(
                self._cur_task, it, self.args["local_ep"], loss_ce_cur.detach(), c_loss_f.detach(), total_ce_loss, total_f_loss, total_kd_loss, total_sf_loss, total_loss))

        return model.state_dict(), total_syn, total_local, total_loss


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
