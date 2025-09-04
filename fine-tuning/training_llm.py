import torch, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

BASE = "meta-llama/Llama-3.2-1B-Instruct" 
USE_QLORA = True  # set False for plain LoRA

# --- Tokenizer ---
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

# --- Dataset ---
def pack(ex):
    sys_, user, ans = ex["system"], ex["user"], ex["assistant"]
    text = f"<s>[INST] <<SYS>>\n{sys_}\n<</SYS>>\n{user} [/INST] {ans} </s>"
    return {"text": text}

raw = load_dataset("json", data_files="data/chatbot_sample.jsonl")["train"]
# Map to a single text field in the Llama [INST] format
packed = raw.map(pack, remove_columns=raw.column_names)
# 70/30 split with a fixed seed for reproducibility
splits = packed.train_test_split(test_size=0.3, seed=42)
train_ds, val_ds = splits["train"], splits["test"]

# --- Model load ---
quant_kwargs = {}
if USE_QLORA:
    from transformers import BitsAndBytesConfig
    quant_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
    )

model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto", trust_remote_code=True, **quant_kwargs
)
model.config.use_cache = False  # better for training

# --- LoRA ---
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]  # add MLP if needed
)
model = get_peft_model(model, lora)

# --- Training args ---
args = TrainingArguments(
    output_dir="lora-host",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    bf16=True,
    report_to="none"
)

# --- Collator ---
def collate(batch):
    out = tok([b["text"] for b in batch], return_tensors="pt",
              padding=True, truncation=True, max_length=1024)
    out["labels"] = out["input_ids"].clone()
    return out

# --- Train ---
trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collate, tokenizer=tok
)
trainer.train()
trainer.save_model("lora-host")
tok.save_pretrained("lora-host")
