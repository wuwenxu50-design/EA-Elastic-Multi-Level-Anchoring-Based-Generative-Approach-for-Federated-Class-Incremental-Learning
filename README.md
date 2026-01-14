

# Method
The EA is a data-free generative method for federated class-incremental learning that aims to curb catastrophic forgetting when new classes arrive over time. It introduces a class-adaptive elastic radius learned from local embedding statistics so each class anchor covers an appropriate high-confidence region under client heterogeneity. It also uses a multi-level anchoring scheme where second-level anchors capture intra-class sub-distributions and then modulate first-level semantic anchors through a FiLM-based generator to produce more representative and diverse synthetic samples. Across CIFAR-100, Stanford Dogs, Tiny-ImageNet, and ImageNet, EA consistently improves average accuracy and reduces forgetting under non-IID splits, with the strongest gains reported on CIFAR-100. 
<p align="center">
  <img src="imgs/General Architecture of EA.png" alt="EA framework" width="700">
</p>

# Reproducing

```
torch==2.0.1
torchvision==0.15.2
```

## Baseline
Here, we provide a simple example for different methods. 
For example, for `cifar100-5tasks`, please run the following commands to test the model performance with non-IID (`$\beta=0.5$`) data.

```
#!/bin/bash
# method= ["finetue", "lwf", "ewc", "icarl", "target", "lander"]

CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=$method_b05 --dataset cifar100 --method=$method --tasks=5 --num_users 5 --beta=0.5
```

### Ours
```
CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=ea_b05 --dataset cifar100 --method=ea --tasks=5 --num_users 5 --beta=0.5
```

## References:
  ```
[1]Zhang J, Chen C, Zhuang W, et al. Target: Federated class-continual learning via exemplar-free
distillation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023:
4782-4793.
[2]Tran M T, Le T, Le X M, et al. Text-enhanced data-free approach for federated class-incremental
learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
2024: 23870-23880.
  ```
