import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def train(args):
    # --- 1. Load the dataset ---
    # The dataset is expected to be in JSONL format with "prompt" and "completion" keys.
    # We combine them into a single "text" field for the trainer.
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"<s>[INST] {example['prompt'][i]} [/INST] {example['completion'][i]} </s>"
            output_texts.append(text)
        return output_texts

    dataset = load_dataset("json", data_files=os.path.join(args.train_dir, "dataset.jsonl"))
    
    # --- 2. Load the tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # --- 3. Configure LoRA (PEFT) ---
    # This makes training much more memory-efficient
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # Specific to Llama architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # --- 4. Set up Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_total_limit=2,
        report_to="tensorboard",
        fp16=False, # Use bf16 if available on the instance
        bf16=True,
    )

    # --- 5. Initialize the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                    'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                    'labels': torch.stack([f['input_ids'] for f in data])}
    )

    # --- 6. Start Training ---
    trainer.train()

    # --- 7. Save the final model ---
    # The trained LoRA adapters will be saved to the specified model_dir
    trainer.save_model(args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker environment variables
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    
    # Model and training parameters
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()
    train(args)
