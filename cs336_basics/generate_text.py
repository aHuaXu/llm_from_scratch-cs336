import torch
from torch import nn
from cs336_basics.bpe_tokenizer import BpeTokenizer
from typing import List
from cs336_basics.base_module import softmax


def generate_text(
    model: nn.Module,
    prompt: torch.Tensor,   # (batch_size, seq_len)
    tokenizer: BpeTokenizer,
    temperature: float = 1.0,  # temperature for softmax sampling
    max_generated_tokens: int = 100,
    end_token: str = "<|endoftext|>",
) -> List[str]:
    """
    Deliverable: Implement a function to decode from your language model.

    We recommend that you support the following features:

    • Generate completions for a user-provided prompt (i.e., take in some x1...t and sample a
      completion until you hit an <|endoftext|> token).
    • Allow the user to control the maximum number of generated tokens.
    • Given a desired temperature value, apply softmax temperature scaling to the predicted
      next-word distributions before sampling.
    • Top-p sampling (Holtzman et al., 2020; also referred to as nucleus sampling), given a
      user-specified threshold value.
    """
    # device = prompt.device
    end_token_id: int = tokenizer.encode(end_token)[0]
    
    model.eval()
    gen_tokens = prompt
    with torch.no_grad():
        for _ in range(max_generated_tokens):
            y = model(gen_tokens)  # y.shape: (batch_size, seq_len, vocab_size)
            y_last = y[:, -1, :]  # y_last.shape: (batch_size, vocab_size)

            y_last = softmax(y_last, dim=-1, temperature=temperature)

            sampled_pos = torch.multinomial(y_last, 1)

            gen_tokens = torch.cat([gen_tokens, sampled_pos], dim=-1)

            # ToDo: implement nucleus or top-p sampling

            if (sampled_pos == end_token_id).all():
                break

    gen_texts: List[str] = []
    for seq in gen_tokens:
        end_pos = (seq == end_token_id).nonzero()
        if len(end_pos) > 0:
            first_pos = end_pos[0].item()
            seq = seq[:first_pos]
        gen_texts.append(tokenizer.decode(seq.tolist()))

    return gen_texts
