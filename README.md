# Skew-PromptKD: Stable Prompt-Based Knowledge Distillation for Generative Language Models via Skew Divergence




> **Skew-PromptKD: Stable Prompt-Based Knowledge Distillation for Generative Language Models via Skew Divergence**<br>
> Marzieh Azad <br>

>**Abstract**: <br>
> Abstract—Prompt-based knowledge distillation (KD) has recently emerged as a lightweight and effective method for com-
pressing large generative language models (LLMs). However,
existing methods such as PROMPTKD rely on reverse KL
divergence, which often suffers from gradient instability and
poor convergence, especially in the early stages of training. In
this work, we propose Skew-PromptKD, a novel distillation
framework that incorporates skew Kullback-Leibler divergence
(SKL) to improve the stability, efficiency, and performance of
prompt-based KD.
Skew-PromptKD preserves the parameter efficiency of soft
prompt tuning while replacing standard divergence objectives
with α-SKL and α-SRKL, which interpolate between teacher and
student distributions to yield smoother optimization dynamics.
We further introduce a regularization loss to stabilize prompt
learning during early training. Additionally, we investigate the
use of hybrid prompting, combining fixed hard prompts with
learned soft prompts to enhance instruction generalization.
We evaluate Skew-PromptKD on five instruction-following
benchmarks using GPT-2 model families. Our method consistently outperforms prior distillation approaches—including
PROMPTKD, MINILLM, and DISTILLM—in terms of
ROUGE-L, while requiring minimal additional parameters. Ab-
lation studies confirm the effectiveness of skew divergence and
hybrid prompting in improving both convergence and general-
ization.

## Requirements

See `install.sh`
```
conda create -n Skew-PromptKD python=3.8
conda activate Skew-PromptKD
bash install.sh
```

### Pretrained models

In order to use this project you need to download pretrained models and datasets. 
(Many of them are from MiniLLM github repository)

Use the `download_requirements.sh` script
```
bash download_requirements.sh
```
This script downloads 
- Model(GPT-2, OPT, Llama) checkpoints for reproducibility in `checkpoints` folder. 
    - Here, we provide PromptKD checkpoints via huggingface hub ([Link](https://huggingface.co/collections/gmkim/promptkd-66728dc78171db46e7fb7bcd))
- Data for training and evaluation in `data` and `processed_data` folder.


## Evaluation

Reproduce the performance of the model checkpoint used in the paper.

```
bash scripts/{model}/eval/run_eval.sh . checkpoints/{model}/train/{method}/{model}-{size}
```
You can choose `{model}` in `{gpt2, opt, llama}`, and `{method}` in `{sft, kd, seqkd, minillm, promptkd, Skew-PromptKD}`.

For GPT-2, you can choose `{size}` in `{base, medium, large}`.  
For OPT, you can choose `{size}` in `{1.3B, 2.7B, 6.7B}`.  
For Llama, put `7B` in `{size}`.


## Training

Training scripts are as below.
```
bash scripts/{model}/{method}/{method}_{student_size}.sh # For training
bash scripts/{model}/eval/run_eval.sh . results/{model}/train/{method}/{model}-{student_size}-{teacher_size}/{your_exp_folder}/best_rougeL # For evaluation
```
You can choose `{model}` in `{gpt2, opt, llama}`, and `{method}` in `{sft, kd, seqkd, minillm, promptkd, Skew-PromptKD}`.

Determine `{student_size}` and `{teacher_size}` by checking the `scripts` folder.  
Also, put your experiment output folder in `{your_exp_folder}` by checking the `results/{model}/train` folder.



### Change Model Parallel Size
You can increase/decrease the tensor parallel sizes with
```
python3 tools/convert_mp.py \
    --input_path checkpoints/llama/train/sft/llama-13B \
    --source_mp_size 4 \
    --target_mp_size 1 \
    --model_type llama # choose from opt and llama
```




---

## 🔬 Skew-PromptKD (New)

Skew-PromptKD improves prompt-based distillation via:

- Skewed KL divergences (`alpha`-SKL, `alpha`-SRKL)
- Hybrid prompting (soft + hard)
- Exposure-bias mitigation with student-generated outputs

### 🔧 Training

```bash
python train_promptkd_skl.py \
  --model_name gpt2 \
  --teacher_name gpt2-xl \
  --output_dir ./promptkd_skl_out \
  --alpha 0.1 \
  --lambda_reg 1.0 \
  --epochs 3
```

### 📈 Evaluation

```bash
python evaluate_skl.py --model_path ./promptkd_skl_out
```

---
## Credits
PromptKD implementation:  
[[https://github.com/microsoft/LMOps/tree/main/minillm](https://promptkd.github.io/)](https://github.com/gmkim-ai/PromptKD)  
License (MIT) https://github.com/microsoft/LMOps/blob/main/LICENSE  
