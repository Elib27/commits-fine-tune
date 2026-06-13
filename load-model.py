import torch
from transformers import (
    Mistral3ForConditionalGeneration,
    AutoTokenizer,
)
# to convert to bf16
# pip install mlx-lm
# python -m mlx_lm.convert \
#     --hf-path mistralai/Ministral-3-3B-Instruct-2512 \
#     --mlx-path ./ministral-3b-bf16-mlx \
#     --dtype bfloat16

# to direclty use MLX
# python -m mlx_lm lora \
#   --model mistralai/Ministral-3-3B-Instruct-2512 \
#   --data ./data \
#   --train \
#   --fine-tune-type lora \
#   --batch-size 4 \
#   --iters 1000 \
#   --adapter-path ./adapters-ministral-commits
#
# model_id = "mistralai/Ministral-3-3B-Instruct-2512"

model_path = "./ministral-3b-bf16-mlx"

# loading the model
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="mps",
)

tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from peft import LoraConfig, PeftModel, get_peft_model

lora_config = LoraConfig(
    r=16,  # Rank — how expressive the adapter is
    lora_alpha=32,  # Scaling factor (usually 2×r)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[  # Which layers to apply LoRA to
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
assert isinstance(model, PeftModel)

model.print_trainable_parameters()
# Output: trainable params: 13,631,488 || all params: 3,766,599,680 || trainable%: 0.36

from datasets import load_dataset
from collections import Counter

dataset = load_dataset("json", data_files="commits.jsonl", split="train")

repo_counts = Counter(dataset["meta"][i]["repo"] for i in range(len(dataset)))
print(repo_counts)


# Apply the chat template to each example
def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


dataset = dataset.shuffle(seed=42)

# Split into train / eval
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

train_repo_counts = Counter(
    train_dataset["meta"][i]["repo"] for i in range(len(train_dataset))
)
eval_repo_counts = Counter(
    eval_dataset["meta"][i]["repo"] for i in range(len(eval_dataset))
)

train_dataset = train_dataset.map(
    format_example,
    remove_columns=["messages", "meta"],  # Drop everything except "text"
)

eval_dataset = eval_dataset.map(
    format_example,
    remove_columns=["messages", "meta"],  # Drop everything except "text"
)

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
print(train_repo_counts)
print(eval_repo_counts)

from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

training_args = SFTConfig(
    output_dir="./ministral-commits",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=100,
    dataloader_pin_memory=False,  # Suppresses the MPS pin_memory warning
    dataloader_num_workers=0,  # Avoids multiprocessing issues on MPS
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    dataset_text_field="text",
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
