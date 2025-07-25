
# evaluate_skl.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
import argparse

def evaluate(model_path, max_gen_len=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()

    dataset = load_dataset("databricks/databricks-dolly-15k", split="test[:1%]")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    total_score = 0
    count = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        input_text = sample["instruction"]
        ref = sample["output"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            pred_ids = model.generate(input_ids, max_length=max_gen_len)
        pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

        score = scorer.score(ref, pred)["rougeL"].fmeasure
        total_score += score
        count += 1

    avg_score = total_score / count
    print(f"Average ROUGE-L: {avg_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./promptkd_skl_out")
    parser.add_argument("--max_gen_len", type=int, default=128)
    args = parser.parse_args()

    evaluate(args.model_path, args.max_gen_len)
