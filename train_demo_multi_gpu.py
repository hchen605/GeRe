import os
import torch
import transformers
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
)
import datasets
from gere.gere_trainer import GeReTrainer
from accelerate import Accelerator
# import ipdb

transformers.utils.logging.set_verbosity(transformers.logging.INFO)


def prepare_dataset(tokenizer, local_path="./yelp_train.json"):
    """ load yelp downstream datasets
    """
    dataset = datasets.load_dataset("json", data_files=local_path, split="train")
    labels_str = ', '.join(["very negative", "negative", "neutral", "positive", "very positive"])
    def tokenize_function(examples):
        sources = [
            f"What is the sentiment of the following paragraph? Choose one from the option.\nOption: {labels_str}\n{sent}\nAnswer:"
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
            attention_mask.append([1] * (len(tokenized_s)+len(tokenized_t)))
            
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



def main():
    # Load model and tokenizer
    model_name = "./base_llms/llama-3.1-tiny-random"  # For test
    # model_name = "llama3.1-8B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    accelerator = Accelerator()
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer)

    # Configure training parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir="./ckpts/test_tiny_ckpts",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=11,
        num_train_epochs=3,
        logging_steps=1,
        bf16=True,
        learning_rate=3e-6,
        lr_scheduler_type='cosine',
        warmup_steps=10,
        remove_unused_columns=False,  # notice!
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        save_strategy="no",  # "steps", "epoch", "no"
        save_only_model=True,
        report_to="none",
        gradient_checkpointing=True,
        deepspeed="ds_config_zero2.json",  # specify DeepSpeed configuration,
    )

    # Default data collator for padding
    # data_collator = DataCollatorWithPadding(tokenizer)
    def get_default_data_collator(tokenizer):
        def data_collator_fn(batch):
            # right-pad 
            padded_inputs = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(e['input_ids']) for e in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_labels = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(e['labels']) for e in batch], batch_first=True, padding_value=-100)
            padded_attention_mask = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(e['attention_mask']) for e in batch], batch_first=True, padding_value=0)

            inputs = {
                "input_ids": padded_inputs,
                "labels": padded_labels,
                "attention_mask": padded_attention_mask,
            }
            # inspect(ret, tokenizer)
            return inputs

        return data_collator_fn
    data_collator = get_default_data_collator(tokenizer)

    # Initialize default Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    # )
    
    # initialize GeRe Trainer 
    trainer = GeReTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # GeRe specific configurations:↓↓↓
        gere_hidden_state_saving_dir='./tiny_gere_hidden_save_dir',  # Dir to save GeRe hidden states and statistics
        reuse_gere_hidden_state=True,  # If False, will force regeneration of hidden states and statistics in the specified directory, 
                                       # but notice existing hidden states will be skipped (Generate missing hidden states and update statistics)
        num_interpolate_per_batch=0,  # BI ratio. set to 0 or None to disable.
        w_strategy='100'  # weight strategy of margin loss. ['1', '100', 'dy'] dy means dynamic
    )
    
    # Prepare model and trainer using Accelerator
    model, trainer = accelerator.prepare(model, trainer)
    
    # Training
    train_result = trainer.train()
    
    # Save results
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

if __name__ == "__main__":
    main()  # this python file should be invoked by the script(train_demo_multi_gpu.sh)
