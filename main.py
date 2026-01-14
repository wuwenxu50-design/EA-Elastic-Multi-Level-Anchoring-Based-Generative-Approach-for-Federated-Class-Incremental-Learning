import argparse
import wandb, os
from utils.data_manager import DataManager, setup_seed
from utils.toolkit import count_parameters
from methods.finetune import Finetune
from methods.icarl import iCaRL
from methods.lwf import LwF
from methods.ewc import EWC
from methods.target import TARGET
from methods.lander import LANDER
from methods.EA import EA
import warnings
import sys
import atexit
warnings.filterwarnings('ignore')
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            try:
                f.flush()
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

def get_learner(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        return iCaRL(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "target":
        return TARGET(args)
    elif name == "lander":
        return LANDER(args)
    elif name == "ea":
        return EA(args)
    else:
        assert 0


def train(args):
    setup_seed(args["seed"])
    # setup the dataset and labels
    data_manager = DataManager(
        args["dataset"],
        args["class_shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args
    )
    args["class_order"] = data_manager.get_class_order()
    learner = get_learner(args["method"], args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}

    # train for each task
    for task in range(data_manager.nb_tasks):
    # for task in range(2):
        print("All params: {}, Trainable params: {}".format(count_parameters(learner._network),
                                                            count_parameters(learner._network, True)))
        learner.incremental_train(data_manager)  # train for one task
        cnn_accy, nme_accy = learner.eval_task()
        learner.after_task()

        print("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        print("CNN top1 curve: {}".format(cnn_curve["top1"]))


def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # Exp settings
    parser.add_argument('--exp_name', type=str, default='ea_b0', help='name of this experiment')
    parser.add_argument('--wandb', type=int, default=0, help='1 for using wandb')
    parser.add_argument('--save_dir', type=str, default="", help='save the syn data')
    parser.add_argument('--project', type=str, default="EA", help='wandb project')
    parser.add_argument('--group', type=str, default="c100", help='wandb group')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--spec', type=str, default="t1", help='choose a model')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    # federated continual learning settings
    parser.add_argument('--dataset', type=str, default="cifar100", help='which dataset')
    parser.add_argument('--tasks', type=int, default=5, help='num of tasks')
    parser.add_argument('--method', type=str, default="EA", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=100, help='communication rounds')
    parser.add_argument('--local_ep', type=int, default=2, help='local training epochs')
    parser.add_argument('--num_users', type=int, default=5, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--beta', type=float, default=1, help='control the degree of label skew')
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')
    parser.add_argument('--class_shuffle', type=int, default=1, help='class shuffle')
    # Long-tailed CIFAR-100 (CIFAR-100-LT)
    #parser.add_argument('--imb_type', type=str, default="none", choices=["none", "exp", "step"],help='long-tail imbalance type for CIFAR-100 train set')
    #parser.add_argument('--imb_factor', type=float, default=1.0,help='imbalance factor (e.g., 0.1/0.02/0.01). smaller => more imbalanced')
    #parser.add_argument('--imb_seed', type=int, default=0,help='random seed for long-tail subsampling')

    # Data-free Generation
    parser.add_argument('--lr_g', default=2e-3, type=float, help='learning rate of generator')
    parser.add_argument('--synthesis_batch_size', default=256, type=int, help='synthetic data batch size')
    parser.add_argument('--bn', default=1.0, type=float, help='parameter for batchnorm regularization')
    parser.add_argument('--oh', default=0.5, type=float, help='parameter for similarity')
    parser.add_argument('--adv', default=1.0, type=float, help='parameter for diversity')
    parser.add_argument('--nz', default=256, type=int, help='output size of noisy nayer')
    parser.add_argument('--nums', type=int, default=10000, help='the num of synthetic data')
    parser.add_argument('--warmup', default=10, type=int, help='number of epoches generator only warmups not stores images')
    parser.add_argument('--syn_round', default=40, type=int, help='number of synthetize round.')
    parser.add_argument('--g_steps', default=40, type=int, help='number of generation steps.')

    # Client Training
    parser.add_argument('--num_worker', type=int, default=4, help='number of worker for dataloader')
    parser.add_argument('--mulc', type=str, default="fork", help='type of multi process for dataloader')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--syn_bs', default=1, type=int, help='number of old synthetic data in training, 1 for similar to local_bs')
    parser.add_argument('--local_lr', default=4e-2, type=float, help='learning rate for optimizer')

    # EA
    parser.add_argument('--r', default=0.015, type=float, help='LTE center radius')
    parser.add_argument('--use_soft_radius', type=int, default=1,help='1: use per-class learnable soft radius; 0: fallback to fixed scalar r')
    parser.add_argument('--r_init', type=float, default=0.015,help='initial radius value (before softplus) for each class')
    parser.add_argument('--lr_r', type=float, default=0.01,help='learning rate for updating soft radius only')
    parser.add_argument('--r_loss_w', type=float, default=1.0,help='weight for updating elastic soft radius via boundary loss')
    parser.add_argument('--r_reg', type=float, default=0.001,help='shrinkage regularizer on radius_hat (encourage compact boundaries)')
    parser.add_argument('--ltc', default=5, type=float, help='lamda_ltc parameter for LTE center')
    parser.add_argument('--pre', type=float, default=0.4, help='alpha_pre for distilling from previous task')
    parser.add_argument('--cur', type=float, default=0.2, help='alpha_cur for current task training')
    parser.add_argument('--k2', dest='K2', type=int, default=3,help='EA: number of secondary anchors per class (K2)')
    parser.add_argument('--type', default=-1, type=int,help='seed for initializing training.') # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2
    parser.add_argument('--syn', default=1, type=int,help='seed for initializing training.')  # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = args_parser()
    if args.dataset == "tiny_imagenet":
        args.num_class = 200
    elif args.dataset == "cifar100":
        args.num_class = 100
    elif args.dataset == "imagenet":
        args.num_class = 1000
    elif args.dataset == "stanford_cars":
        args.num_class = 196
    args.init_cls = int(args.num_class / args.tasks)
    args.increment = args.init_cls

    print(args)

    args.exp_name = f"{args.beta}_{args.method}_{args.exp_name}"

    dir = "run"
    if not os.path.exists(dir):
        os.makedirs(dir)
    args.save_dir = os.path.join(dir, args.group + "_" + args.exp_name + "" + args.spec)

    if args.wandb == 1:
        wandb.init(config=args, project=args.project, group=args.group, name=args.exp_name)
        wandb.run.log_code(".")
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_root, "forgetting_calculator", "log")
    os.makedirs(log_dir, exist_ok=True)
    beta_tag = str(args.beta).replace(".", "")
    if beta_tag == "0":
        beta_tag = "00"
    log_name = f"{args.method}_t{args.tasks}b{beta_tag}_seed{args.seed}.log"  
    log_path = os.path.join(log_dir, log_name)
    _stdout = sys.stdout
    _stderr = sys.stderr
    log_f = open(log_path, "a", encoding="utf-8")
    atexit.register(log_f.close)
    sys.stdout = Tee(_stdout, log_f)
    sys.stderr = Tee(_stderr, log_f)
    print(f"[LOG] Saving run logs to: {log_path}")

    args = vars(args)
    train(args)

