import os
import logging
import sys
from tqdm import tqdm
import glob
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
import random
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..model import BERT, BERTLM

os.environ["HF_HOME"] = "/scratch/xtong/cache/"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/xtong/cache/"
logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more verbosity
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

WARMUP_PROPORTION = 0.1
GRADIENT_CLIP_NORM = 0.5
MIN_LEARNING_RATE = 1e-6


class BERTTrainer:
    """
    BERTTrainer pretrains BERT model with two LM training method:
    1. Masked Language Modelling
    2. Sentence Order prediction or Next Sentence Prediction

    """

    def __init__(
        self,
        bert: BERT,
        tokenizer=None,
        dataloader=None,
        epochs=None,
        optimizer=None,
        wandb_log=True,
        vocab_size=None,
        embed_dim=None,
        n_layers=None,
        num_heads=None,
        max_length=None,
        ff_dropout=None,
        load_checkpoint_path=None,
        save_checkpoint_path=None,
        language=None,
        dataset=None,
        lr=None,
        use_scheduler=None,
        input_id=None,
        batch_size=None,
        special_condition=None
    ):

        self.global_step = 0
        self.scaler = torch.cuda.amp.GradScaler()
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduce=True)
        self.wandb_log = wandb_log
        self.epochs = epochs
        self.checkpoint_path = save_checkpoint_path
        self.id = input_id
        if not torch.cuda.is_available():
            logger.info("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.max_length = max_length
        self.dataset = dataset
        self.use_scheduler = use_scheduler
        self.lr = lr
        self.ff_dropout = ff_dropout
        self.special_condition = special_condition
        
        if load_checkpoint_path is not None and os.path.isfile(load_checkpoint_path):
            logger.info(f"Loading checkpoint from {load_checkpoint_path}")
            checkpoint = torch.load(load_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            logger.info("Model state loaded from checkpoint.")
            self.embed_dim = checkpoint.get("embed_dim", embed_dim)
            self.n_layers = checkpoint.get("n_layers", n_layers)
            self.num_heads = checkpoint.get("num_heads", num_heads)
            self.vocab_size = checkpoint.get("vocab_size", len(self.tokenizer))
            self.global_step = checkpoint.get("global_step", 0)
            if "optimizer_state" in checkpoint:
                self.optimizer = checkpoint["optimizer_state"]
                print("Optimizer state loaded from checkpoint.")
        else:
            self.bert = bert
            self.vocab_size = vocab_size
            assert self.vocab_size == len(self.tokenizer)
            logger.info(self.vocab_size)
            self.model = BERTLM(bert, self.vocab_size)
            if special_condition == "adam":
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.98),
                    eps=1e-12
                    )
            elif special_condition == "adamw_wd":
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.98),
                    eps=1e-12,
                    weight_decay=0.01,
                    )
            elif special_condition == "adam_lrd":
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=MIN_LEARNING_RATE,
                    betas=(0.9, 0.98),
                    eps=1e-12,
                    )
                
                total_steps = len(dataloader) * epochs
                self.warmup_steps = int(total_steps * WARMUP_PROPORTION)
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, 
                    T_max=total_steps - self.warmup_steps,
                    eta_min=1e-6
                )
            elif special_condition == "adam_wd_lrd":
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=MIN_LEARNING_RATE,
                    betas=(0.9, 0.98),
                    eps=1e-12,
                    weight_decay=0.01,
                    )
                total_steps = len(dataloader) * epochs
                self.warmup_steps = int(total_steps * WARMUP_PROPORTION)
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, 
                    T_max=total_steps - self.warmup_steps,
                    eta_min=1e-6
                )
            elif special_condition == "adamw_wd_lrd":
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=MIN_LEARNING_RATE,
                    betas=(0.9, 0.98),
                    eps=1e-12,
                    weight_decay=0.01,
                    )
                
                total_steps = len(dataloader) * epochs
                self.warmup_steps = int(total_steps * WARMUP_PROPORTION)
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, 
                    T_max=total_steps - self.warmup_steps,  # Steps after warm-up
                    eta_min=1e-6  # Minimum learning rate
                )
        

        if self.wandb_log:
            wandb.init(
                project=f"{language}_mlmsop",
                name=f"{dataset}_{max_length}_{n_layers}_{self.id}_{special_condition}" if special_condition else f"{dataset}_{max_length}_{n_layers}_{self.id}",
                config={
                    "learning_rate": (
                        self.optimizer.param_groups[0]["lr"] if self.optimizer else None
                    ),
                    "architecture": "Transformer",
                    "epochs": epochs,
                    "embed_dim": embed_dim,
                    "n_layers": n_layers,
                    "num_heads": num_heads,
                    "vocab_size": self.vocab_size,
                    "batch_size": batch_size,
                    "ff_dropout": self.ff_dropout,
                    "ckpt_path": self.checkpoint_path,
                    "optimizer":self.optimizer,
                    },
            )
        logger.info(
            f"Total Parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        
    def get_warmup_lr(self, step):
        """
        Computes a linear learning rate for the warm-up phase.

        Args:
            step (int): Current global training step.

        Returns:
            float: Linearly scaled learning rate based on the step and maximum warm-up learning rate.
        """
        max_lr = self.lr
        return max_lr * (step / self.warmup_steps)
    
    def train(self):
        """
        Initiates the training loop, iterating over the entire dataset for the specified number of epochs.
        """
        self.iteration(self.dataloader)

    def iteration(self, dataloader):
        """
        Manages the training loop across epochs and batches, including logging, learning rate scheduling,
        and checkpointing.

        Args:
            dataloader (DataLoader): PyTorch DataLoader providing training batches.
        """
        logger.info("Starting training...")
        self.model.to(self.device)
        self.global_step = getattr(self, "global_step", 0)
        self.scaler = torch.cuda.amp.GradScaler()
        total_sample = len(dataloader.dataset)

        try:
            for epoch in range(self.epochs):
                self.model.train()
                batch_count = 0

                logger.info(f"\nStarting epoch {epoch + 1}/{self.epochs}")

                with tqdm(total=len(dataloader)) as pbar:
                    for batch in dataloader:
                        # Train one batch
                        loss, mlm_logits, sop_logits, mlm_labels, sop_labels = (
                            self.train_one_batch(batch)
                        )

                        # Calculate metrics
                        mlm_acc, sop_acc = self.calculate_metrics(
                            mlm_logits, sop_logits, mlm_labels, sop_labels
                        )

                        # Log metrics
                        self.log_metrics(loss, mlm_acc, sop_acc, epoch)

                        if (
                            self.global_step == 0 or self.global_step % 10000 == 0
                        ):  # Save at first step and every 10000 steps
                            try:
                                self.save_checkpoint(f"iteration_{self.global_step}")
                                logger.info(
                                    f"\nSuccessfully saved checkpoint at step {self.global_step}"
                                )
                            except Exception as e:
                                logger.info(
                                    f"\nFailed to save checkpoint at step {self.global_step}: {e}"
                                )
                                raise
                        if self.special_condition == "adamw_wd_lrd" or self.special_condition == "adam_wd_lrd":
                            if self.global_step < self.warmup_steps:
                                lr = self.get_warmup_lr(self.global_step)
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] = lr
                            else:
                                self.scheduler.step()
                            
                        self.global_step += 1
                        batch_count += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "batch": f"{batch_count}/{total_sample // dataloader.batch_size}",
                                "loss": f"{loss:.4f}",
                            }
                        )
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)

        # Save final checkpoint
        self.save_checkpoint("final")

    def train_one_batch(self, batch):
        """
        Performs training on a single batch, including forward and backward passes, gradient clipping,
        and optimizer step.

        Args:
            batch (dict): Dictionary containing tensors for input data and labels.

        Returns:
            tuple: Loss value and predictions/logits for MLM and SOP tasks, and the corresponding labels.
        """
        input_ids = batch["input_ids"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        mlm_labels = batch["mlm_labels"].to(self.device)
        sop_labels = batch["sop_label"].to(self.device)

        self.optimizer.zero_grad()
        try:
            with torch.cuda.amp.autocast():  # Mixed precision training
                mlm_logits, sop_logits = self.model(
                    input_ids, token_type_ids, attention_mask.unsqueeze(1)
                )
                mlm_loss = self.criterion(
                    mlm_logits.view(-1, self.vocab_size), mlm_labels.view(-1)
                )
                sop_loss = self.criterion(sop_logits, sop_labels)
                loss = mlm_loss + sop_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)  
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        except RuntimeError as e:
            logger.error(
                f"Error during training: {e}, {self.vocab_size, mlm_labels.size(), sop_labels.size(), mlm_logits.size(), sop_logits.size()}"
            )
            torch.cuda.empty_cache()
            raise

        return loss.item(), mlm_logits, sop_logits, mlm_labels, sop_labels

    def calculate_metrics(self, mlm_logits, sop_logits, mlm_labels, sop_labels):
        """
        Calculates accuracy metrics for Masked Language Modeling (MLM) and Sentence Order Prediction (SOP).

        Args:
            mlm_logits (torch.Tensor): Predicted logits for masked language modeling.
            sop_logits (torch.Tensor): Predicted logits for sentence order prediction.
            mlm_labels (torch.Tensor): True labels for MLM task.
            sop_labels (torch.Tensor): True labels for sentence order prediction.

        Returns:
            tuple: MLM accuracy, SOP accuracy.
        """
        # MLM Accuracy
        mlm_pred = mlm_logits.argmax(dim=-1)
        mlm_counted = torch.sum(mlm_labels != -100)
        mlm_correct = torch.sum(mlm_pred == mlm_labels)
        mlm_acc = mlm_correct.item() / mlm_counted.item()
        # SOP/NSP Accuracy (Although they both are named sop_...)
        sop_pred = sop_logits.argmax(dim=-1)
        sop_acc = (sop_pred == sop_labels).float().mean().item()

        return mlm_acc, sop_acc

    def log_metrics(self, loss, mlm_acc, sop_acc, epoch):
        """
        Logs training metrics, such as loss and accuracy, to the console and optionally to wandb.

        Args:
            loss (float): Loss value of the current training step.
            mlm_acc (float): Accuracy of the Masked Language Modeling task.
            sop_acc (float): Accuracy of the Sentence Order Prediction task.
            epoch (int): Current epoch number.
        """
        logger.info(
            f"Step {self.global_step}: Loss = {loss:.4f}, MLM Acc = {mlm_acc:.4f}, SOP/NSP Acc = {sop_acc:.4f}"
        )

        if self.wandb_log and self.global_step % 10 == 0:
            wandb.log(
                {
                    "train_loss": loss,
                    "mlm_accuracy": mlm_acc,
                    "sop/nsp_accuracy": sop_acc,
                    "epoch": epoch + 1,
                    "global_step": self.global_step,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

    def save_checkpoint(self, iteration):
        """
        Saves the current model checkpoint to disk. It includes model weights, optimizer state,
        and training metadata.

        Args:
            iteration (str or int): Identifier for the checkpoint (e.g., 'final' or iteration number).
        """
        checkpoint = {
            "global_step": self.global_step,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "n_layers": self.n_layers,
            "num_heads": self.num_heads,
            "state_dict": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

        
        # Save the new checkpoint
        os.makedirs(self.checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_path,
            f"{self.dataset}_maxlen{self.max_length}_layer{self.n_layers}_{self.id}",
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_final_path = os.path.join(
            checkpoint_path,
            f"checkpoint_{iteration}.pth",
        )
        torch.save(checkpoint, checkpoint_final_path)

        if isinstance(iteration, str) and iteration.startswith("iteration_"):
            checkpoints = glob.glob(
                os.path.join(
                    checkpoint_path,
                    f"checkpoint_iteration_[0-9]*.pth",
                )
            )
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for checkpoint_file in checkpoints[
                :-2
            ]:  # Remove all but the last 2 checkpoints
                os.remove(checkpoint_file)
