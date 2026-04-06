import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 smoke test: load a local/open-source causal LM and run a minimal generation check."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype to use when loading the model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Device to place the model on.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the model repository if required.",
    )
    parser.add_argument(
        "--prompt",
        default="Please briefly assess this claim: Apple will release a quantum computer next year.",
        help="Prompt used for the minimal generation test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate in the smoke test.",
    )
    parser.add_argument(
        "--text-a",
        default="The claim appears credible based on the available evidence.",
        help="First text used for similarity/disagreement testing.",
    )
    parser.add_argument(
        "--text-b",
        default="The claim seems trustworthy according to the reported facts.",
        help="Second text used for similarity/disagreement testing.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def resolve_device(device_name: str) -> str:
    if device_name == "cpu":
        return "cpu"
    if device_name == "mps":
        return "mps"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_text_to_pooled_vector(model, tokenizer, text: str, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    final_hidden = outputs.hidden_states[-1]
    pooled_vector = final_hidden.mean(dim=1)
    return pooled_vector


def main() -> None:
    args = parse_args()
    torch_dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)

    print("=== Phase 1 Load Test ===")
    print(f"Model: {args.model}")
    print(f"Requested dtype: {args.dtype}")
    print(f"Resolved device: {device}")
    print(f"Trust remote code: {args.trust_remote_code}")

    model_source = Path(args.model) if Path(args.model).exists() else args.model

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    first_param = next(model.parameters())
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Parameter dtype: {first_param.dtype}")
    print(f"Parameter device: {first_param.device}")

    print("\nLoad test passed.")

    print("\n=== Minimal Generation Test ===")
    print(f"Prompt: {args.prompt}")

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    new_tokens = generated.shape[1] - input_length

    print(f"Input tokens: {input_length}")
    print(f"Generated new tokens: {new_tokens}")
    print("\nGenerated text:")
    print(generated_text)
    print("\nGeneration test passed.")

    print("\n=== Hidden State Extraction Test ===")
    attention_mask = torch.ones_like(generated, device=generated.device)
    with torch.no_grad():
        outputs = model(
            input_ids=generated,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    final_hidden = outputs.hidden_states[-1]
    if new_tokens > 0:
        pooled_source = final_hidden[:, input_length:, :]
    else:
        pooled_source = final_hidden
    pooled_vector = pooled_source.mean(dim=1)

    print(f"Final hidden states shape: {tuple(final_hidden.shape)}")
    print(f"Pooled vector shape: {tuple(pooled_vector.shape)}")
    print("\nHidden state extraction test passed.")

    print("\n=== Similarity / Disagreement Test ===")
    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")

    pooled_a = encode_text_to_pooled_vector(model, tokenizer, args.text_a, device)
    pooled_b = encode_text_to_pooled_vector(model, tokenizer, args.text_b, device)

    similarity = F.cosine_similarity(pooled_a, pooled_b, dim=1).item()
    disagreement = 1.0 - similarity

    print(f"Pooled vector A shape: {tuple(pooled_a.shape)}")
    print(f"Pooled vector B shape: {tuple(pooled_b.shape)}")
    print(f"Cosine similarity: {similarity:.6f}")
    print(f"Disagreement: {disagreement:.6f}")
    print("\nSimilarity/disagreement test passed.")


if __name__ == "__main__":
    main()
