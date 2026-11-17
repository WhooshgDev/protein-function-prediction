from typing import List, Optional
import numpy as np
import torch


def mean_pool_residues(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean-pool only valid amino acid residue positions (excluding BOS, EOS, and padding).

    Args:
        hidden_states: Tensor of shape (B, L, D)
        attention_mask: Tensor of shape (B, L) where 1 = non-pad, 0 = pad
        special_tokens_mask: Optional Tensor of shape (B, L) where 1 = special token (BOS/EOS/PAD), 0 = residue
    Returns:
        Pooled embeddings of shape (B, D)
    """
    if special_tokens_mask is not None:
        valid_mask = (attention_mask > 0) & (special_tokens_mask == 0)
    else:
        # Standard ESM tokenizer layout: index 0 is BOS (<cls>), last non-pad is EOS (<eos>)
        valid_mask = attention_mask.clone().bool()
        valid_mask[:, 0] = False
        lengths = attention_mask.sum(dim=1)
        for i, length in enumerate(lengths):
            if length > 1:
                valid_mask[i, length - 1] = False

    mask_float = valid_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask_float).sum(dim=1)
    counts = mask_float.sum(dim=1).clamp(min=1.0)
    return summed / counts


def encode_sequences(
    sequences: List[str],
    protein_ids: List[str],
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    batch_size: int = 16,
    max_length: int = 1022,
    max_length_policy: str = "truncate",
    device: Optional[str] = None,
) -> np.ndarray:
    """Contract entrypoint: Extract frozen ESM-2 residue mean-pooled embeddings (N, 640)."""
    from transformers import AutoModel, AutoTokenizer

    if len(sequences) != len(protein_ids):
        raise ValueError("sequences and protein_ids must have identical length.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    all_embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start : start + batch_size]
            if max_length_policy == "truncate":
                batch_seqs = [s[:max_length] for s in batch_seqs]
            else:
                raise ValueError(f"Unsupported max_length_policy: {max_length_policy}")

            encoded = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=max_length + 2,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            special_tokens_mask = encoded.get("special_tokens_mask")
            if special_tokens_mask is not None:
                special_tokens_mask = special_tokens_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pool_residues(
                outputs.last_hidden_state,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
            )
            emb_np = pooled.cpu().to(torch.float32).numpy()
            if not np.isfinite(emb_np).all():
                raise FloatingPointError("Non-finite values detected in ESM embeddings.")
            all_embeddings.append(emb_np)

    result = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 640), dtype=np.float32)
    return result
