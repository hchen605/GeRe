#!/usr/bin/env python3
import os, json, argparse, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from gere import GeReTrainer  # from Qznan/GeRe
from transformers import Trainer, TrainingArguments

# -------------------------
# 15-task long-sequence orders (Progressive Prompts style)
# -------------------------
ORDERS = {
    4: ["mnli","cb","wic","copa","qqp","boolq","rte","imdb","yelp","amazon","sst2","dbpedia_14","ag_news","super_glue.multirc","yahoo_answers_topics"],
    5: ["super_glue.multirc","boolq","wic","mnli","cb","copa","qqp","rte","imdb","sst2","dbpedia_14","ag_news","yelp","amazon","yahoo_answers_topics"],
    6: ["yelp","amazon","mnli","cb","copa","qqp","rte","imdb","sst2","dbpedia_14","ag_news","yahoo_answers_topics","super_glue.multirc","boolq","wic"],
    7: ["yelp","amazon","mnli","cb"]
}

# Map short names to HF datasets/configs
DATASET_SPEC = {
    "mnli": ("glue","mnli"),
    "qqp": ("glue","qqp"),
    "rte": ("glue","rte"),
    "sst2": ("glue","sst2"),
    "cb": ("super_glue","cb"),
    "wic": ("super_glue","wic"),
    "copa": ("super_glue","copa"),
    "boolq": ("super_glue","boolq"),
    "super_glue.multirc": ("super_glue","multirc"),
    "imdb": ("imdb", None),
    "dbpedia_14": ("dbpedia_14", None),
    "ag_news": ("ag_news", None),
    "yelp": ("yelp_polarity", None),
    "amazon": ("amazon_polarity", None),
    "yahoo_answers_topics": ("yahoo_answers_topics", None),
}

# -------------------------
# Prompting & labels (simple versions; for paper-parity, port PP templates)
# -------------------------
def build_prompt(example, task):
    if task in ["sst2","yelp","amazon","imdb"]:
        text = example.get("sentence") or example.get("text") or example.get("content") or example.get("document") or ""
        return f"Classify the sentiment, and only answer one word positive or negative:\n{text}\nAnswer:"
    if task in ["ag_news","dbpedia_14","yahoo_answers_topics"]:
        text = example.get("text") or example.get("content") or (example.get("question_title","") + "\n" + example.get("question_content",""))
        return f"Classify the topic:\n{text}\nAnswer:"
    if task == "mnli":
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nIs the hypothesis entailed, contradicted, or neutral? (only answer one of these: entailment, neutral, or contradiction)\n Answer:"
    if task == "qqp":
        return f"Question1: {example['question1']}\nQuestion2: {example['question2']}\nAre they paraphrases? (yes/no)\nAnswer:"
    if task == "rte":
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nIs the hypothesis entailed? (yes/no)\nAnswer:"
    if task == "cb":
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nChoose: entailment/contradiction/neutral\nAnswer:"
    if task == "wic":
        return f"Word Sense Disambiguation:\nSentence1: {example['sentence1']}\nSentence2: {example['sentence2']}\nWord: {example['word']}\nSame meaning? (yes/no)\nAnswer:"
    if task == "copa":
        return f"Premise: {example['premise']}\nChoice1: {example['choice1']}\nChoice2: {example['choice2']}\nWhich is more plausible? (1/2)\nAnswer:"
    if task == "boolq":
        return f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer yes or no:\nAnswer:"
    if task == "super_glue.multirc":
        return f"Passage: {example['paragraph']}\nQuestion: {example['question']}\nCandidate Answer: {example['answer']}\nIs the candidate correct? (yes/no)\nAnswer:"
    return str(example)

def label_to_text(task, example):
    m = {
        "sst2": {0:"negative",1:"positive"},
        "yelp": {0:"negative",1:"positive"},
        "amazon": {0:"negative",1:"positive"},
        "imdb": {0:"negative",1:"positive"},
        "ag_news": {0:"World",1:"Sports",2:"Business",3:"Sci/Tech"},
        "dbpedia_14": {i: str(i) for i in range(14)},      # placeholder class names
        "yahoo_answers_topics": {i: str(i) for i in range(10)},
        "mnli": {0:"entailment",1:"neutral",2:"contradiction"},
        "qqp": {0:"no",1:"yes"},
        "rte": {0:"not_entailment",1:"entailment"},
        "cb": {0:"entailment",1:"contradiction",2:"neutral"},
        "wic": {0:"no",1:"yes"},
        "copa": {0:"1",1:"2"},
        "boolq": {0:"no",1:"yes"},
        "super_glue.multirc": {0:"no",1:"yes"},
    }
    key = "label" if "label" in example else ("label_coarse" if "label_coarse" in example else None)
    if key is None:
        return None
    return m[task].get(int(example[key]), None)

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def subsample_split(ds, per_class=1000, seed=42, label_key="label"):
    random.seed(seed)
    if label_key not in ds.features:
        return ds.shuffle(seed=seed).select(range(min(len(ds), per_class)))
    byc, idxs = {}, []
    for i, ex in enumerate(ds):
        c = int(ex.get(label_key, 0))
        if byc.get(c, 0) < per_class:
            byc[c] = byc.get(c, 0) + 1
            idxs.append(i)
    return ds.select(idxs)

def make_dataset(task, tokenizer, split="train"):
    from datasets import load_dataset
    name, config = DATASET_SPEC[task]
    ds = load_dataset(name, config, split=split)
    if split == "train":
        label_key = "label" if "label" in ds.features else "label_coarse" if "label_coarse" in ds.features else None
        ds = subsample_split(ds, per_class=1000, seed=42, label_key=label_key or "label")

    def tok_map(ex):
        prompt = build_prompt(ex, task)
        ans = label_to_text(task, ex)
        s_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if ans is None: ans = ""
        t_ids = tokenizer(str(ans), add_special_tokens=False)["input_ids"]
        s_ids = [tokenizer.bos_token_id] + s_ids[:1023]
        t_ids = (t_ids + [tokenizer.eos_token_id])[:32]
        return {
            "input_ids": s_ids + t_ids,
            "labels": [-100]*len(s_ids) + t_ids,
            "attention_mask": [1]*(len(s_ids)+len(t_ids)),
        }

    ds = ds.map(tok_map, batched=False, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    return ds

def make_collator(tokenizer):
    def collate(batch):
        pad_id = tokenizer.pad_token_id
        to = lambda xs, v: torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x) for x in xs], batch_first=True, padding_value=v)
        return {
            "input_ids": to([b["input_ids"] for b in batch], pad_id),
            "labels": to([b["labels"] for b in batch], -100),
            "attention_mask": to([b["attention_mask"] for b in batch], 0),
        }
    return collate

@torch.inference_mode()
def eval_task_accuracy(model, dataloader, device):
    """
    Exact-match accuracy over the answer span using teacher forcing.
    We compare argmax(logits) at t to the gold token at t (after shifting).
    A sample is counted correct only if *all* answer tokens match.
    """
    model.eval()
    total, correct = 0, 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # SHIFT for causal LM: logits predict the *next* token
        logits = outputs.logits[:, :-1, :]                 # [B, T-1, V]
        labels = batch["labels"][:, 1:]                    # [B, T-1]
        target_mask = (labels != -100)                     # only answer tokens are supervised

        if target_mask.any():
            preds = logits.argmax(dim=-1)                  # [B, T-1]
            # equality only where supervised
            eq = (preds == labels) | (~target_mask)
            # a sample is correct if *all* supervised positions are correct
            sample_correct = eq.all(dim=1)
            correct += sample_correct.sum().item()
            total += sample_correct.shape[0]
    return (correct / total) if total > 0 else 0.0

def compute_cl_metrics(acc_matrix):
    """
    acc_matrix[i][t] = accuracy on task i measured after finishing training task t.
    """
    acc = np.array(acc_matrix)  # [N, N]
    N = acc.shape[0]
    AA = float(acc[:, -1].mean())
    AF = 0.0
    for i in range(N):
        seen = acc[i, i:]  # from time learned to end
        AF += (float(seen.max()) - float(seen[-1]))
    AF /= N
    BWT = float(np.mean([acc[i, -1] - acc[i, i] for i in range(N-1)])) if N > 1 else 0.0
    return {"AA": AA, "AF": AF, "BWT": BWT}

def _decode_prompt_and_gold(batch_item_ids, batch_item_labels, tokenizer):
    """
    From a single sample's tensors, recover the prompt text and the gold answer text.
    We assume labels == -100 for prompt tokens and labels contain answer tokens.
    """
    # Find first label position that is supervised (start of answer)
    lbl = batch_item_labels
    ans_pos = (lbl != -100).nonzero(as_tuple=False)
    if ans_pos.numel() == 0:
        # No supervised span; treat all as prompt
        return tokenizer.decode(batch_item_ids, skip_special_tokens=True), ""
    s = int(ans_pos[0])
    prompt_ids = batch_item_ids[:s]
    gold_ids = lbl[lbl != -100]
    prompt_txt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    gold_txt = tokenizer.decode(gold_ids, skip_special_tokens=True)
    return prompt_txt, gold_txt


@torch.inference_mode()
def eval_and_maybe_log_jsonl(
    model,
    tokenizer,
    dataloader,
    device,
    max_new_tokens=4,
    save_path: str | None = None,
    save_limit: int = 0,  # 0 = no cap
):
    """
    Greedy-generate short answers; compute accuracy using:
        ok = (norm(pred) == norm(gold)) or (gold != "" and gold in pred)
    If save_path is provided, also dump JSONL rows {idx, prompt, gold, pred, correct}.
    Returns: (accuracy_float, num_rows_saved_int)
    """
    model.eval()
    total = 0
    correct = 0
    saved = 0

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = open(save_path, "w", encoding="utf-8")
    idx_global = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # find prompt-only cut (first supervised label position)
        ans_starts = []
        for i in range(labels.size(0)):
            pos = (labels[i] != -100).nonzero(as_tuple=False)
            ans_starts.append(int(pos[0]) if pos.numel() > 0 else input_ids.size(1))

        # build prompt batch (truncate to answer start)
        prompts = []
        for i, s in enumerate(ans_starts):
            s = max(1, min(s, input_ids.size(1)))
            prompts.append(input_ids[i, :s])

        pad_id = tokenizer.pad_token_id
        max_len = max(p.size(0) for p in prompts)
        def pad_to(t, L):
            if t.size(0) == L: return t
            return torch.cat([t, t.new_full((L - t.size(0),), pad_id)], dim=0)
        prompt_batch = torch.stack([pad_to(p, max_len) for p in prompts], dim=0)

        # greedy-generate short answers
        gen = model.generate(
            input_ids=prompt_batch,
            attention_mask=(prompt_batch != pad_id).long(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        preds = tokenizer.batch_decode(gen[:, max_len:], skip_special_tokens=True)

        # compare + (optionally) log rows
        for i, pred_txt in enumerate(preds):
            prompt_txt, gold_txt = _decode_prompt_and_gold(
                input_ids[i].cpu(), labels[i].cpu(), tokenizer
            )
            gnorm = " ".join(gold_txt.lower().split())
            pnorm = " ".join(pred_txt.lower().split())
            ok = (pnorm == gnorm) or (gnorm != "" and gnorm in pnorm)

            correct += int(ok)
            total += 1

            if writer:
                row = {
                    "idx": idx_global,
                    "prompt": prompt_txt,
                    "gold": gold_txt,
                    "pred": pred_txt.strip(),
                    "correct": bool(ok),
                }
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                saved += 1
                idx_global += 1
                if save_limit and saved >= save_limit:
                    writer.close()
                    return (correct / total if total else 0.0, saved)

    if writer:
        writer.close()
    return (correct / total if total else 0.0, saved)


@torch.inference_mode()
def save_generations_jsonl(model, tokenizer, dataloader, out_path, max_rows=0, max_new_tokens=8, device="cuda:0"):
    """
    Greedy-generate from the prompt (up to the answer start) and save JSONL:
    {idx, prompt, gold, pred, correct}
    """
    import json
    model.eval()
    written = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        idx_global = 0
        for batch in dataloader:
            # move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Build prompt-only inputs per sample
            ans_starts = []
            for i in range(labels.size(0)):
                pos = (labels[i] != -100).nonzero(as_tuple=False)
                ans_starts.append(int(pos[0]) if pos.numel() > 0 else input_ids.size(1))

            prompts = []
            for i, s in enumerate(ans_starts):
                s = max(1, min(s, input_ids.size(1)))  # guard
                prompts.append(input_ids[i, :s])

            # Left-pad to uniform length for batching
            pad_id = tokenizer.pad_token_id
            max_len = max(p.size(0) for p in prompts)
            def pad_to(t, L):
                if t.size(0) == L: return t
                pad = t.new_full((L - t.size(0),), pad_id)
                return torch.cat([t, pad], dim=0)
            prompt_batch = torch.stack([pad_to(p, max_len) for p in prompts], dim=0)

            gen = model.generate(
                input_ids=prompt_batch,
                attention_mask=(prompt_batch != pad_id).long(),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_tail = gen[:, max_len:]
            preds = tokenizer.batch_decode(gen_tail, skip_special_tokens=True)

            # decode prompt+gold; compute correctness (case-insensitive containment or exact)
            for i in range(input_ids.size(0)):
                prompt_txt, gold_txt = _decode_prompt_and_gold(input_ids[i].cpu(), labels[i].cpu(), tokenizer)
                pred_txt = preds[i].strip()
                gnorm = " ".join(gold_txt.lower().split())
                pnorm = " ".join(pred_txt.lower().split())
                correct = (pnorm == gnorm) or (gnorm != "" and gnorm in pnorm)

                row = {
                    "idx": idx_global,
                    "prompt": prompt_txt,
                    "gold": gold_txt,
                    "pred": pred_txt,
                    "correct": bool(correct),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                idx_global += 1

                if max_rows and written >= max_rows:
                    return written
    return written

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=4, choices=[4,5,6,7])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output_root", default="./ckpts/gere_lora_15tasks")
    ap.add_argument("--eval_batch_size", type=int, default=8)
    # GeRe knobs (set BI to >0 to mimic "with BI" line; choose w_strategy per ablation)
    ap.add_argument("--num_interpolate_per_batch", type=int, default=0)
    ap.add_argument("--w_strategy", type=str, default="100", choices=["1","100","dy"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_generations_dir", type=str, default="", help="If set, dump JSONL of (prompt, gold, pred, correct) per eval.")
    ap.add_argument("--save_generations_max", type=int, default=200, help="Max rows to save per (step, task). 0=all.")
    ap.add_argument("--gen_max_new_tokens", type=int, default=8, help="Max new tokens for generation when saving outputs.")
    ap.add_argument("--ft", default="gere")
    args = ap.parse_args()

    set_seed(args.seed)

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id

    base = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda:0", low_cpu_mem_usage=True)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
        target_modules=["q_proj","k_proj"], task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base, lora_cfg)
    data_collator = make_collator(tokenizer)

    tasks = ORDERS[args.order]
    N = len(tasks)
    acc_matrix = [[0.0]*N for _ in range(N)]

    # cache eval sets/dataloaders
    cached_eval = {}
    def get_eval_loader(task_name):
        if task_name not in cached_eval:
            try:
                ds_eval = make_dataset(task_name, tokenizer, split="validation")
            except Exception:
                ds_eval = make_dataset(task_name, tokenizer, split="test")
            cached_eval[task_name] = DataLoader(ds_eval, batch_size=args.eval_batch_size, shuffle=False, collate_fn=data_collator)
        return cached_eval[task_name]

    model.train()  # ensure training mode
    if True:  # if you pass gradient_checkpointing=True in args
        # Trainer will call gradient_checkpointing_enable(), but being explicit helps with PEFT
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        # REQUIRED with checkpointing: disable key/value caching or the graph gets cut
        if hasattr(model, "config"):
            model.config.use_cache = False
        # Some transformer wrappers benefit from this to keep inputs requiring grad
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tensors_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}  (~{trainable/total*100:.3f}%)")
    print(f"Trainable tensors: {tensors_trainable}")
    # num_trainable = sum(p.requires_grad for p in model.parameters())
    # print('num_trainable: ', num_trainable)
    # assert num_trainable > 0, "No trainable parameters detected — LoRA may not be attached."


    # Train sequentially; after each task, re-evaluate all seen tasks
    for t_idx, t in enumerate(tasks):
        ds_train = make_dataset(t, tokenizer, split="train")
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{args.output_root}/{t}",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,  # global batch ~64
            num_train_epochs=8,
            logging_steps=10,
            bf16=True,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=True,
            save_strategy="no",
            report_to="none",
            gradient_checkpointing=True,
        )

        # >>> Plain Trainer (no GeRe)
        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=ds_train,
        #     data_collator=data_collator,
        #     tokenizer=tokenizer,
        # )

        trainer = GeReTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            data_collator=data_collator,
            tokenizer=tokenizer,
            # GeRe-specific configs
            gere_hidden_state_saving_dir=f'./hidden/llama31-8b/{t}',
            reuse_gere_hidden_state=True,
            num_interpolate_per_batch=args.num_interpolate_per_batch,  # BI: set to 4 for global batch=64 (paper)
            w_strategy=args.w_strategy
        )

        trainer.train()

        # Re-evaluate all tasks seen so far
        device = model.device if hasattr(model, "device") else "cuda:0"
        for i in range(t_idx+1):
            ti = tasks[i]
            print('Evaluating... ', ti)
            loader = get_eval_loader(ti)
            
            outfile = None
            if args.save_generations_dir:
                outfile = os.path.join(
                    args.save_generations_dir,
                    f"{args.ft}_order{args.order}_after_t{t_idx+1:02d}_{ti}.jsonl"
                )

            acc, n_saved = eval_and_maybe_log_jsonl(
                model=model,
                tokenizer=tokenizer,
                dataloader=loader,
                device=device,
                max_new_tokens=args.gen_max_new_tokens,
                save_path=outfile,
                save_limit=max(0, args.save_generations_max),
            )
            acc_matrix[i][t_idx] = float(acc)
            msg = f"[order {args.order}] after {t_idx+1}/{N} ({t}), eval on {ti}: {acc:.4f}"
            if outfile:
                msg += f"  -> saved {n_saved} rows to {outfile}"
            print(msg)


    # Compute CL metrics
    cl = compute_cl_metrics(acc_matrix)
    print(f"CL metrics — AA={cl['AA']:.4f}  AF={cl['AF']:.4f}  BWT={cl['BWT']:.4f}")

    # Save JSON + CSV
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    out_json = f"{args.output_root}/acc_matrix_order{args.order}.json"
    with open(out_json, "w") as f:
        json.dump({"order": args.order, "tasks": tasks, "acc_matrix": acc_matrix, "metrics": cl}, f, indent=2)
    out_csv = f"{args.output_root}/acc_matrix_order{args.order}.csv"
    with open(out_csv, "w") as f:
        # header: t0 ... tN-1
        f.write("task," + ",".join([f"after_t{t}" for t in range(N)]) + "\n")
        for i, name in enumerate(tasks):
            f.write(name + "," + ",".join(f"{acc_matrix[i][t]:.4f}" for t in range(N)) + "\n")
        f.write(f"# AA,{cl['AA']:.4f}\n# AF,{cl['AF']:.4f}\n# BWT,{cl['BWT']:.4f}\n")
    print(f"Saved: {out_json}\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
