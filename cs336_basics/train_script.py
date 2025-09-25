import argparse
import logging
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb

from cs336_basics.train_transformer import (
    cross_entropy_loss,
    save_checkpoint,
    load_checkpoint,
)
from cs336_basics.transformer import (
    TransformerLM,
)
from cs336_basics.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
)
from cs336_basics.data import (
    get_batch,
    DatasetForTransformer,
)
from cs336_basics.helper import (
    get_current_datetime,
    project_name,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train.log'),
    ],
)
logger = logging.getLogger(__name__)

def train_one_epoch(
    model: nn.Module,
    opt: optim.Optimizer,
    data_loader: DataLoader,
    max_l2_norm: float,
    epoch: int,
    global_step: int,
) -> (float, int):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(data_loader):
        global_step += 1

        # ToDo: use lr_cosine_schedule
        # lr = get_lr_cosine_schedule(global_step, max_l2_norm)

        logits = model(x)
        opt.zero_grad()

        vocab_size = logits.shape[-1]
        loss = cross_entropy_loss(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm)

        opt.step()
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(
                f"Batch {batch_idx}/{len(data_loader)}"
                f"Loss: {avg_loss:.4f}"
                f"Time: {time.time() - start_time:.2f}s")
            wandb.log({
                "batch_loss": loss.item(),
                "avg_batch_loss": avg_loss,
                "epoch": epoch + 1,
                "step": global_step,
                "wallclock_time": time.time() - start_time
            })

    avg_epoch_loss = total_loss / len(data_loader)
    return avg_epoch_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer model')

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--vocab_size', type=int, required=True, help='vocab size')
    parser.add_argument('--context_length', type=int, required=True, help='context length')
    parser.add_argument("--num_layers", type=int, default=2, help='number of layers')
    parser.add_argument("--d_model", type=int, default=512, help='hidden dimension')
    parser.add_argument("--d_ff", type=int, default=2048, help='hidden dimension')
    parser.add_argument("--rope_theta", type=float, default=0.1, help='rope_theta')
    parser.add_argument("--data_path", type=str, default='./data', help='data path')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint', help='checkpoint path')
    parser.add_argument("--max_l2_norm", type=float, default=1000, help='max l2 norm')

    args = parser.parse_args()
    return args

def main(args: argparse.Namespace):
    wandb.init(
        project=project_name,
        name="transformer" + get_current_datetime(),
        config=vars(args),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": str(device)})

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
        dtype=torch.float32,
        rope_theta=args.rope_theta,
    )
    wandb.watch(model, log="all")

    opt = AdamW(model.parameters(), lr=args.lr)

    train_dataset = DatasetForTransformer(
        data_path=args.data_path,
        context_length=args.context_length,
        device=device,
        dtype=torch.float32,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    start_epoch = 0
    if args.checkpoint_path:
        trained_epoch = load_checkpoint(args.checkpoint_path, model, opt)
        start_epoch = trained_epoch + 1

    experiment_start_time = time.time()
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_loss, global_step = train_one_epoch(model, opt, train_loader,
                args.max_l2_norm, epoch, global_step)

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
        wandb.log({
            "epoch_loss": train_loss,
            "epoch_time": epoch_time,
            "epoch": epoch + 1,
            "global_step": global_step,
            "wallclock_time": time.time() - experiment_start_time
        })
        save_checkpoint(model, opt, args.checkpoint_path, epoch)

    wandb.finish()
    logger.info(f"Total training finished: {time.time() - experiment_start_time:.2f}s")

if __name__ == "__main__":
    args = parse_args()
    main(args)