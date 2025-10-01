import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
import datasets
from gere import GeReTrainer  # from the repo

transformers.utils.logging.set_verbosity(transformers.logging.INFO)


def prepare_dataset(tokenizer, local_path="./yelp_train.json"):
    """Load yelp downstream dataset and tokenize for causal LM."""
    dataset = datasets.load_dataset("json", data_files=local_path, split="train")
    labels_str = ', '.join(["very negative", "negative", "neutral", "positive", "very positive"])

    def tokenize_function(examples):
        sources = [
            f"What is the sentiment of the following paragraph? Choose one from the option.\n"
            f"Option: {labels_str}\n{sent}\nAnswer:"
            for sent in examples['sentence']
        ]
        targets = examples['label']

        input_ids, labels, attention_mask = [], [], []
        for s, t in zip(sources, targets):
            tokenized_s = tokenizer(s, add_special_tokens=False)["input_ids"]
            tokenized_t = tokenizer(t, add_special_tokens=False)["input_ids"]

            tokenized_s = [tokenizer.bos_token_id] + tokenized_s[:511]
            tokenized_t = tokenized_t + [tokenizer.eos_token_id]
            tokenized_t = tokenized_t[:50]

            input_ids.append(tokenized_s + tokenized_t)
            labels.append([-100] * len(tokenized_s) + tokenized_t)
            attention_mask.append([1] * (len(tokenized_s) + len(tokenized_t)))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def get_default_data_collator(tokenizer):
    def data_collator_fn(batch):
        # right-pad
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(e['input_ids']) for e in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(e['labels']) for e in batch],
            batch_first=True,
            padding_value=-100
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(e['attention_mask']) for e in batch],
            batch_first=True,
            padding_value=0
        )
        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": padded_attention_mask,
        }
    return data_collator_fn


def main():
    # --- Base model / tokenizer ---
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )

    # --- LoRA config (as per paper) ---
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    # (optional but helpful) show trainable param count
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # --- Data ---
    dataset = prepare_dataset(tokenizer)
    data_collator = get_default_data_collator(tokenizer)

    # Effective global batch ~= 64 (1 * 64 grad acc)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./ckpts/llama-31-8b-instruct-lora-yelp",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        num_train_epochs=3,             # LoRA: 8 epochs (paper)
        logging_steps=1,
        bf16=True,
        learning_rate=1e-4,             # LoRA LR from paper
        lr_scheduler_type='cosine',
        warmup_steps=10,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        save_strategy="no",
        save_only_model=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    # --- GeRe Trainer (unchanged API) ---
    trainer = GeReTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # GeRe-specific configs (same as your demo; adjust as needed)
        gere_hidden_state_saving_dir='./hidden/llama-31-8b-instruct_hidden_save_dir',
        reuse_gere_hidden_state=True,
        num_interpolate_per_batch=0,   # set >0 to enable BI ratio
        w_strategy='100'               # ['1','100','dy'] choose per your ablation
    )

    train_result = trainer.train()

    # Save LoRA adapters (and trainer state/metrics)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)


if __name__ == "__main__":
    main()
