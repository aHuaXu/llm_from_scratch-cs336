import argparse
from datetime import datetime
import logging
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb

from .train_transformer import (
    cross_entropy_loss,
    save_checkpoint,
    load_checkpoint,
)
from .transformer import (
    TransformerLM,
)
from .optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
)
from .data import (
    get_batch,
    DatasetForTransformer,
)
from .train_bpe import (
    train_bpe,
    default_chunk_generator,
    get_vocab_merges_path,
    load_vocab_and_merges,
)
from .bpe_tokenizer import BpeTokenizer
import numpy as np
from tqdm import tqdm
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log"),
    ],
)
logger = logging.getLogger(__name__)

project_name = "cs336_assignment1"

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def prepare_memmap_data(vocab_size=10000, dtype=np.int32) -> None:
    # 1. 路径设置
    train_data_path = "./data/TinyStoriesV2-GPT4-train.txt"
    valid_data_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    output_train_bin = "./data/train_tokens.bin"  # 二进制文件
    output_valid_bin = "./data/valid_tokens.bin"

    if os.path.exists(output_train_bin) and os.path.exists(output_valid_bin):
        return

    # 2. 训练BPE并初始化分词器
    vocab_path, merges_path = get_vocab_merges_path(train_data_path)
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        vocabs, merges = load_vocab_and_merges(vocab_path, merges_path)
    else:
        vocabs, merges = train_bpe(
            input_path=train_data_path,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
        )
    tokenizer = BpeTokenizer(vocabs, merges, special_tokens=["<|endoftext|>"])

    # 3. 定义二进制写入函数
    def process_to_binary(input_path: str, output_path: str) -> None:
        """将文本编码为token ID，并以二进制格式写入（支持memmap读取）"""
        chunk_num = 1000
        chunk_generator = default_chunk_generator(input_path, chunk_num=chunk_num)

        if os.path.exists(output_path):
            print(f"{output_path} already exists, skipping.")
            return

        print(f"start to encode into {output_path}")
        # 以二进制写模式打开文件
        with open(output_path, "ab") as f:
            with tqdm(
                total=chunk_num, desc=f"Encoding {input_path}", unit="chunk"
            ) as pbar:
                for chunk in chunk_generator:
                    # 编码为token ID列表
                    token_ids = tokenizer.encode(chunk)
                    if not token_ids:
                        continue
                    token_array = np.array(token_ids, dtype=dtype)
                    token_array.tofile(f)

                    pbar.update(1)

        print(f"encode into {output_path} finished")

    # 4. 处理训练集和验证集
    process_to_binary(train_data_path, output_train_bin)
    process_to_binary(valid_data_path, output_valid_bin)


# def train_one_iteration(
#     model: nn.Module,
#     opt: optim.Optimizer,
#     data_loader: DataLoader,
#     max_l2_norm: float,
#     epoch: int,
#     global_step: int,
# ) -> (float, int):
#     model.train()
#     total_loss = 0.0
#     start_time = time.time()
#
#     for batch_idx, (x, y) in enumerate(data_loader):
#         global_step += 1
#
#         # ToDo: use lr_cosine_schedule
#         # lr = get_lr_cosine_schedule(global_step, max_l2_norm)
#
#         logits = model(x)
#         opt.zero_grad()
#
#         vocab_size = logits.shape[-1]
#         loss = cross_entropy_loss(logits.reshape(-1, vocab_size), y.reshape(-1))
#         loss.backward()
#         gradient_clipping(model.parameters(), max_l2_norm)
#
#         opt.step()
#         if batch_idx % 100 == 0 and batch_idx > 0:
#             avg_loss = total_loss / (batch_idx + 1)
#             logger.info(
#                 f"Batch {batch_idx}/{len(data_loader)}"
#                 f"Loss: {avg_loss:.4f}"
#                 f"Time: {time.time() - start_time:.2f}s"
#             )
#             wandb.log(
#                 {
#                     "batch_loss": loss.item(),
#                     "avg_batch_loss": avg_loss,
#                     "epoch": epoch + 1,
#                     "step": global_step,
#                     "wallclock_time": time.time() - start_time,
#                 }
#             )
#
#     avg_epoch_loss = total_loss / len(data_loader)
#     return avg_epoch_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train transformer model")

    parser.add_argument("--total_iteration", type=int, default=800, help="total number of iterations")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocab size")
    parser.add_argument("--context_length", type=int, default=256, help="context length")

    parser.add_argument("--num_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--num_heads", type=int, default=16, help="number of heads")
    parser.add_argument("--d_model", type=int, default=512, help="hidden dimension")
    parser.add_argument("--d_ff", type=int, default=1344, help="hidden dimension")
    parser.add_argument("--rope_theta", type=float, default=10000, help="rope_theta")

    parser.add_argument("--data_path", type=str, default="./data/train_tokens.bin", help="data path")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--checkpoint_path", type=str, default="./data/train.ckpt", help="checkpoint path")
    parser.add_argument("--max_l2_norm", type=float, default=10, help="max l2 norm")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    prepare_memmap_data(args.vocab_size)

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

    # train_dataset = DatasetForTransformer(
    #     data_path=args.data_path,
    #     context_length=args.context_length,
    #     device=device,
    #     dtype=torch.float32,
    # )
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=2,
    # )
    memmap_data = np.memmap(args.data_path, mode="r", dtype=np.int32)
    train_dataset = torch.from_numpy(memmap_data).to(dtype=torch.long, device=device)

    start_iter = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        trained_iter = load_checkpoint(args.checkpoint_path, model, opt)
        start_iter = trained_iter + 1

    experiment_start_time = time.time()

    model.train()
    for cur_iter in range(start_iter, args.total_iteration):
        iter_start_time = time.time()

        # ToDo: use lr_cosine_schedule
        # lr = get_lr_cosine_schedule(global_step, max_l2_norm)

        x, y = get_batch(train_dataset, args.batch_size, args.context_length, device)
        logits = model(x)
        opt.zero_grad()

        vocab_size = logits.shape[-1]
        loss = cross_entropy_loss(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)

        opt.step()

        iter_time = time.time() - iter_start_time
        logger.info(
            f"Iter: {cur_iter+1}/{args.total_iteration} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Time: {iter_time:.2f}s"
        )
        wandb.log(
            {
                "loss": loss.item(),
                "iter_time": iter_time,
                "iteration": cur_iter+1,
                "wallclock_time": time.time() - experiment_start_time,
            }
        )
        if cur_iter%5 == 0:
            save_checkpoint(model, opt, cur_iter, args.checkpoint_path)

    wandb.finish()
    logger.info(f"Total training finished: {time.time() - experiment_start_time:.2f}s")


if __name__ == "__main__":
    args = parse_args()
    main(args)
