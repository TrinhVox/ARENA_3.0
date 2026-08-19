import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from part3_optimization.common import WandbResNetFinetuningArgs, get_cifar
from part3_optimization.part1_dist_training.all_reduce import all_reduce
from part3_optimization.part1_dist_training.broadcast import broadcast



device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

def get_untrained_resnet(n_classes: int) -> ResNet34:
    """
    Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your
    implementation).
    """
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


class DistResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")

    def pre_training_setup(self):
        self.model = get_untrained_resnet(self.args.n_classes).to(self.device)
        if self.args.world_size > 1:
            for param in self.model.parameters():
                broadcast(param.data, self.rank, self.args.world_size, src=0)
        self.optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.trainset, self.testset = get_cifar()

        
    
        
        self.examples_seen = 0
        self.train_sampler = self.test_sampler = None
        dataloader_shared_kwargs = dict(batch_size=self.args.batch_size, num_workers=2, pin_memory=True)
        if self.args.world_size > 1:
            self.train_sampler = DistributedSampler(self.trainset, num_replicas=self.args.world_size, rank=self.rank)
            self.test_sampler = DistributedSampler(self.testset, num_replicas=self.args.world_size, rank=self.rank)
        self.train_loader = DataLoader(self.trainset, sampler=self.train_sampler, **dataloader_shared_kwargs)
        self.test_loader = DataLoader(self.testset, sampler=self.test_sampler, **dataloader_shared_kwargs)

        if self.rank == 0:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_name,
                config=self.args,
            )


    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        """
        Performs a single training step on the given batch of images and labels.
        """
        t0 = time.time()
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        
        outputs = self.model(imgs)
        t1 = time.time()


        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        t2 = time.time()

        if self.args.world_size > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    all_reduce(param.grad, self.rank, self.args.world_size, op="mean")
        t3 = time.time()

        self.optim.step()
        self.optim.zero_grad()
        self.examples_seen += len(imgs) * self.args.world_size  # Count the total number of examples seen across all ranks
        if self.rank == 0:
            wandb.log(
                {"loss": loss.item(), 
                 "examples_seen": self.examples_seen,
                "forward_time": t1 - t0,
                 "backward_time": t2 - t1,
                 "optimization_time": t3 - t2
                })
        return loss
    
    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluates the model on the test set and returns the accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        for imgs, labels in self.test_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outputs = self.model(imgs)
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        tensor = t.tensor([correct, total], device=self.device)
        all_reduce(tensor, self.rank, self.args.world_size, op="sum")
        total_correct, total_samples = tensor.tolist()

        accuracy = total_correct / total_samples
        if self.rank == 0:
            wandb.log({"accuracy": accuracy}, step=self.examples_seen)
        
        return accuracy
    
    def train(self):
        self.pre_training_setup()
        accuracy = self.evaluate()
        for epoch in range(self.args.epochs):
            t0 = time.time()
            if self.args.world_size > 1:
                self.train_sampler.set_epoch(epoch)  # Shuffle the data differently at each epoch
                self.test_sampler.set_epoch(epoch)
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training", disable=self.rank != 0)
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen=:06}")

            accuracy = self.evaluate()

            if self.rank == 0:
                wandb.log({"epoch_duration": time.time() - t0}, step=self.examples_seen)
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    accuracy=f"{accuracy:.3f}",
                    ex_seen=f"{self.examples_seen=:06}",
                )

        if self.rank == 0:
            wandb.finish()
            t.save(self.model.state_dict(), f"resnet_{self.rank}.pth")


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


if MAIN:
    print("Downloading CIFAR (single process)...")
    get_cifar()          # downloads and extracts; runs once, in the parent
    print("Done.")

    world_size = t.cuda.device_count()
    mp.spawn(
        dist_train_resnet_from_scratch,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
