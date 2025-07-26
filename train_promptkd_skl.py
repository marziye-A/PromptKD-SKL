
# train_promptkd_skl.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from datasets import load_dataset
import argparse
import os
from tqdm import tqdm

# ---------------------------
# Skew KL utilities
# ---------------------------

def alpha_skl(p, q, alpha=0.1, eps=1e-8):
    p = p + eps
    q = q + eps
    mix = alpha * p + (1 - alpha) * q
    return torch.sum(p * (torch.log(p) - torch.log(mix)), dim=-1).mean()

def alpha_srkl(q, p, alpha=0.1, eps=1e-8):
    p = p + eps
    q = q + eps
    mix = alpha * q + (1 - alpha) * p
    return torch.sum(q * (torch.log(q) - torch.log(mix)), dim=-1).mean()

# ---------------------------
# Hybrid Prompt Encoder
# ---------------------------

def encode_hybrid_prompt(tokenizer, soft_prompt_embeds, hard_prompt_text, input_text, model):
    input_ids = tokenizer.encode(hard_prompt_text + input_text, return_tensors="pt").to(model.device)
    hard_prompt_len = len(tokenizer.encode(hard_prompt_text))
    inputs_embeds = model.transformer.wte(input_ids)
    soft_prompt_embeds = soft_prompt_embeds.unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1)
    inputs_embeds = torch.cat([inputs_embeds[:, :hard_prompt_len, :], soft_prompt_embeds, inputs_embeds[:, hard_prompt_len:, :]], dim=1)
    return inputs_embeds, input_ids

# ---------------------------
# Main Training Loop
# ---------------------------

def train_promptkd_skl(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    teacher = GPT2LMHeadModel.from_pretrained(args.teacher_name).to(device)
    student = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:1%]")  # Use small split for demo
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    soft_prompt = nn.Parameter(torch.randn(args.soft_tokens, student.config.n_embd)).to(device)
    optimizer = AdamW(list(student.parameters()) + [soft_prompt], lr=args.lr)

    teacher.eval()
    hard_prompt = args.hard_prompt

    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            instructions = batch["instruction"]
            loss_total = 0

            for inst in instructions:
                # Stage 1: Generate pseudo-targets from student
                inputs = tokenizer(inst, return_tensors="pt").to(device)
                with torch.no_grad():
                    pseudo_output = student.generate(inputs["input_ids"], max_length=args.max_length)

                # Stage 2: Prompt-tune teacher to match pseudo-output
                prompt_inputs, _ = encode_hybrid_prompt(tokenizer, soft_prompt, hard_prompt, inst, teacher)
                with torch.no_grad():
                    plain_outputs = teacher(**inputs, labels=pseudo_output)

                prompted_outputs = teacher(inputs_embeds=prompt_inputs, labels=pseudo_output)
                p_prompt = torch.softmax(prompted_outputs.logits, dim=-1)
                q_student = torch.softmax(student(**inputs).logits, dim=-1)

                loss_prompt = alpha_skl(p_prompt, q_student, alpha=args.alpha)
                loss_reg = nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(p_prompt, dim=-1), torch.log_softmax(plain_outputs.logits, dim=-1))

                # Stage 3: Distill into student using skewed reverse KL
                student_outputs = student(**inputs)
                p_teacher = torch.softmax(prompted_outputs.logits, dim=-1)
                q_student = torch.softmax(student_outputs.logits, dim=-1)

                loss_student = alpha_srkl(q_student, p_teacher, alpha=args.alpha)

                # Combine all losses
                total_loss = loss_prompt + args.lambda_reg * loss_reg + loss_student
                loss_total += total_loss.item()

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            print(f"Avg Loss: {loss_total / len(instructions):.4f}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Student model saved to {args.output_dir}")

# ---------------------------
# Argparse
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--teacher_name", type=str, default="gpt2-xl")
    parser.add_argument("--output_dir", type=str, default="./promptkd_skl_out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--soft_tokens", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--hard_prompt", type=str, default="Think like a student-friendly teacher and answer: ")
    args = parser.parse_args()

    train_promptkd_skl(args)
