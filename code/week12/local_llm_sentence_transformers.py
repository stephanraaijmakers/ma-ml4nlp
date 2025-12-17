import argparse
import os
from typing import List, Tuple
# Stephan Raaijmakers, 2025


# pip install sentence_transformers
# Sample run: python local_llm_sentence_transformers.py --model google/gemma-2-2b-it --input_file input.txt --max_new_tokens 30

# Sample input.txt:
# -------------------
#Prompt: What is an empathetic version of: I don't care about that. Respond with only 1 example (a single sentence). 
#I'm sorry but I have little affinity with that
#Prompt: What is a sarcastic version of: I care about that. Respond with only 1 example (a single sentence). 
#I don't care at all

# You can try out prompt designs like this:
#python local_llm_sentence_transformers.py --model google/gemma-2-2b-it --question "What is an empathetic version of: I don't care about that. Respond with just one sentence (a rephrase)" --reference "I'm sorry but I have little affinity with that" --max_new_tokens 20


import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name_or_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """Load a (small) causal LLM such as Gemma locally using transformers."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading model '{model_name_or_path}' on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Use float16 on GPU, float32 on CPU for compatibility.
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    question: str,
    max_new_tokens: int = 128,
) -> str:
    """Generate an answer from the local LLM for the given question."""
    # Very simple prompt format; adapt if you want a different style.
    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the full sequence and strip the prompt part if possible.
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Try to cut off the prompt prefix if it matches.
    if full_text.startswith(prompt):
        answer = full_text[len(prompt) :].strip()
    else:
        answer = full_text.strip()

    return answer


def cosine_similarity_embeddings(reference: str, hypothesis: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    """Compute cosine similarity between reference and hypothesis sentence embeddings."""
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode([reference, hypothesis])
    ref_emb, hyp_emb = embeddings[0], embeddings[1]
    # Cosine similarity
    num = float(np.dot(ref_emb, hyp_emb))
    den = float(np.linalg.norm(ref_emb) * np.linalg.norm(hyp_emb))
    if den == 0.0:
        return 0.0
    return num / den


def read_prompt_reference_pairs(path: str) -> List[Tuple[str, str]]:
    """Read blocks of two lines: 'Prompt: ...' and reference answer.

    Expected format per example (no blank lines required):
      Prompt: <question text>
      <reference answer>
    """
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i + 1 < len(lines):
        prompt_line = lines[i].strip()
        ref_line = lines[i + 1].strip()
        if not prompt_line.startswith("Prompt:"):
            raise ValueError(f"Expected line starting with 'Prompt:' at line {i+1}, got: {prompt_line!r}")
        question = prompt_line[len("Prompt:") :].strip()
        reference = ref_line
        pairs.append((question, reference))
        i += 2

    if i < len(lines):
        # Odd number of lines -> last line has no pair
        raise ValueError("Input file has an odd number of lines; each example must have exactly two lines.")

    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a local LLM (e.g., Gemma), generate an answer for a question, "
            "and evaluate it against a reference answer using cosine similarity on sentence embeddings."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help=(
            "Hugging Face model ID or local path to a causal LLM. "
            "Default is a small Gemma instruction-tuned model."
        ),
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=False,
        help=(
            "Optional path to a file containing multiple examples as two-line blocks: "
            "'Prompt: <text>' on the first line and the reference answer on the second line. "
            "If not provided, you must specify --question and --reference."
        ),
    )

    parser.add_argument(
        "--question",
        type=str,
        required=False,
        help="Input question / prompt to ask the model (single-example mode).",
    )

    parser.add_argument(
        "--reference",
        type=str,
        required=False,
        help="Reference (gold) answer string for evaluation (single-example mode).",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate for the answer.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model)

    # If an input file is provided, process all examples and write predictions.
    if args.input_file:
        pairs = read_prompt_reference_pairs(args.input_file)
        generic_out_path = os.path.join(os.path.dirname(args.input_file) or ".", "predictions.txt")

        print(f"Processing {len(pairs)} examples from {args.input_file} ...")
        print(f"Writing predictions to {generic_out_path}")

        with open(generic_out_path, "w", encoding="utf-8") as out_f:
            for idx, (question, reference) in enumerate(pairs, start=1):
                hypothesis = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                )

                cosine_sim = cosine_similarity_embeddings(reference=reference, hypothesis=hypothesis)

                out_f.write(f"Example {idx}\n")
                out_f.write(f"Prompt: {question}\n")
                out_f.write(f"Reference: {reference}\n")
                out_f.write(f"Answer: {hypothesis}\n")
                out_f.write(f"CosineSimilarity: {cosine_sim:.4f}\n")
                out_f.write("\n")

        print("Done.")

    else:
        # Single example mode: require question and reference.
        if not args.question or not args.reference:
            raise SystemExit("Either provide --input_file or both --question and --reference.")

        print("\n=== QUESTION ===")
        print(args.question)

        print("\n=== REFERENCE ANSWER ===")
        print(args.reference)

        print("\nGenerating model answer...\n")
        hypothesis = generate_answer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
        )

        print("=== MODEL ANSWER ===")
        print(hypothesis)

        print("\nComputing cosine similarity between reference and model answer (sentence embeddings)...\n")
        cosine_sim = cosine_similarity_embeddings(reference=args.reference, hypothesis=hypothesis)

        print("=== COSINE SIMILARITY (SENTENCE EMBEDDINGS) ===")
        print(f"cosine_similarity: {cosine_sim:.4f}")


if __name__ == "__main__":
    main()
