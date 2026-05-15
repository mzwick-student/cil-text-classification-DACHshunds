import torch
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def get_error_dataframe(model, dataloader, tokenizer, device="cuda"):
    model.eval()
    results = []
    
    # Ensure the model is on the correct device
    model.to(device)
    
    for batch in tqdm(dataloader, desc="Calculating Errors"):
        # Move required tensors to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Capture indices (passed through from the dataset)
        # If 'row_idx' isn't in batch, it falls back to a range (less safe)
        row_indices = batch.get("row_idx", torch.arange(len(labels)))

        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Using view(-1) is safer than squeeze() for batch_size=1
            preds = outputs.get("logits").view(-1)
            
            # Calculate absolute error
            errors = torch.abs(preds - labels.view(-1))
            
        # Move to CPU once for faster iteration
        preds_cpu = preds.cpu().numpy()
        errors_cpu = errors.cpu().numpy()
        labels_cpu = labels.view(-1).cpu().numpy()
        indices_cpu = row_indices.cpu().numpy() if torch.is_tensor(row_indices) else row_indices

        # Decode for manual inspection (optional, but helpful for debugging)
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Handle language mapping
        langs = batch.get("lang", [0] * len(texts))
        if torch.is_tensor(langs):
            langs = langs.cpu().numpy()

        for i in range(len(texts)):
            results.append({
                "row_idx": int(indices_cpu[i]),
                "text": texts[i], # Will be truncated to 128 tokens
                "true_label": float(labels_cpu[i]),
                "predicted": float(preds_cpu[i]),
                "error": float(errors_cpu[i]),
                "lang": "eng_Latn" if langs[i] == 0 else "deu_Latn"
            })
                
    return pd.DataFrame(results)

# Run it:
# df = get_error_dataframe(model, trainer.get_eval_dataloader(), tokenizer)

def plot_error_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=df, 
        x="true_label", 
        y="error", 
        hue="lang", 
        split=True, 
        inner="quart"
    )
    plt.title("Absolute Error Distribution by Class and Language")
    plt.xlabel("True Rating (0-4)")
    plt.ylabel("Absolute Error")
    plt.show()

# plot_error_distribution(df)

def get_worst_offenders(df, lang=None, true_label=None, top_n=20):
    filtered_df = df.copy()
    
    if lang:
        filtered_df = filtered_df[filtered_df["lang"] == lang]
    if true_label is not None:
        filtered_df = filtered_df[filtered_df["true_label"] == true_label]
        
    return filtered_df.sort_values(by="error", ascending=False).head(top_n)

# Example: Look at the 10 worst errors for German texts that were actually 4-star
# worst_de_4 = get_worst_offenders(df, lang="deu_Latn", true_label=4.0, top_n=10)
# print(worst_de_4[["predicted", "error", "text"]])

def calculate_pruned_mae(df, n_exclude=0):
    """Calculates the MAE after removing the top N highest-error samples."""
    if n_exclude == 0:
        return df["error"].mean()
    
    # Sort descending and slice off the worst offenders
    pruned_df = df.sort_values(by="error", ascending=False).iloc[n_exclude:]
    
    original_mae = df["error"].mean()
    new_mae = pruned_df["error"].mean()
    
    improvement = original_mae - new_mae
    pct_improvement = (improvement / original_mae) * 100
    
    print(f"Original MAE: {original_mae:.4f}")
    print(f"Pruned MAE (excluding worst {n_exclude}): {new_mae:.4f}")
    print(f"Improvement: {improvement:.4f} ({pct_improvement:.2f}%)")
    
    return new_mae, pruned_df


import os

def save_custom_model(model, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save the weights
    torch.save(model.state_dict(), os.path.join(save_directory, "model_weights.bin"))
    
    # Save the config (if it's a standard HF config)
    model.config.save_pretrained(save_directory)
    
    print(f"Model saved to {save_directory}")

def load_custom_model(save_directory, device="cuda"):
    # 1. Load the configuration
    # This ensures the architecture parameters (hidden size, etc.) match
    config = XLMRobertaConfig.from_pretrained(save_directory)
    
    # 2. Re-initialize the model architecture
    # You must have the XLMRobertaSentimentRegressor class defined in your script
    model = XLMRobertaSentimentRegressor("xlm-roberta-base", config)
    
    # 3. Load the weights into the architecture
    weights_path = f"{save_directory}/model_weights.bin"
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval() # Set to evaluation mode for inference
    
    print(f"Model successfully loaded from {save_directory}")
    return model

def get_full_dataset_errors(model, dataloader, tokenizer, device="cuda"):
    model.eval()
    model.to(device)
    results = []
    
    for batch in tqdm(dataloader, desc="Scanning Full Dataset"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device) # Note: 'label', not 'labels'
        row_indices = batch["row_idx"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use .view(-1) to handle batches of 1 safely
            preds = outputs.get("logits").view(-1)
            errors = torch.abs(preds - labels.view(-1))
            
        # Move to CPU in bulk
        preds_cpu = preds.cpu().numpy()
        errors_cpu = errors.cpu().numpy()
        labels_cpu = labels.view(-1).cpu().numpy()
        indices_cpu = row_indices.numpy() # row_idx is already on CPU if not sent to device

        for i in range(len(indices_cpu)):
            results.append({
                "row_idx": int(indices_cpu[i]),
                "error": float(errors_cpu[i]),
                "predicted": float(preds_cpu[i]),
                "true_label": float(labels_cpu[i])
            })
                
    return pd.DataFrame(results)

def prune_and_save_dataset(hf_dataset, error_df, n_to_prune, output_path="train_lang_pruned.csv"):
    # Convert HF Dataset to Pandas for the .isin() operation
    df = hf_dataset.to_pandas()
    
    # Identify and remove the worst samples
    worst_texts = set(error_df.sort_values(by="error", ascending=False).head(n_to_prune)["text"])
    pruned_df = df[~df["sentence"].isin(worst_texts)]
    
    # Save for future use
    pruned_df.to_csv(output_path, index=False)
    print(f"Pruned {len(df) - len(pruned_df)} samples. New file saved to {output_path}")
    return pruned_df

# Usage
# Ensure original_df is the one loaded via load_dataset
#prune_and_save_dataset(original_df, df_errors, 7500, output_path="train_lang_pruned.csv")

# --- Usage Example ---
# 1. Get the error data
# df_errors = get_error_dataframe(model, trainer.get_train_dataloader(), tokenizer)

# 2. Prune top 500 worst offenders from your original loaded CSV
# df_pruned = prune_and_save_dataset(df_train_original, df_errors, n_to_prune=500)

def plot_pruning_impact(df, max_exclude_pct=0.05):
    """Plots the MAE drop as you exclude up to 'max_exclude_pct' of the data."""
    total_samples = len(df)
    max_exclude = int(total_samples * max_exclude_pct)
    
    sorted_df = df.sort_values(by="error", ascending=False)
    
    ns = list(range(0, max_exclude, max(1, max_exclude // 50)))
    maes = []
    
    for n in ns:
        if n == 0:
            maes.append(sorted_df["error"].mean())
        else:
            maes.append(sorted_df.iloc[n:]["error"].mean())
            
    plt.figure(figsize=(10, 5))
    plt.plot(ns, maes, marker='o', markersize=4, linestyle='-', color='red')
    
    # Formatting
    plt.title(f"The 'Garbage' Curve: MAE vs Excluded Samples (Top {max_exclude_pct*100:.1f}%)")
    plt.xlabel(f"Number of Worst Samples Excluded (out of {total_samples})")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate the start and end points
    plt.annotate(f"{maes[0]:.4f}", (ns[0], maes[0]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f"{maes[-1]:.4f}", (ns[-1], maes[-1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.show()

import torch
import torch.nn.functional as F

def check_interference_sampled(trainer, model, num_batches=10):
    model.train()
    device = next(model.parameters()).device
    dataloader = trainer.get_train_dataloader()
    similarities = []
    
    def get_grads(sub_batch):
        model.zero_grad()
        
        # Get raw outputs (logits)
        outputs = model(
            input_ids=sub_batch["input_ids"], 
            attention_mask=sub_batch["attention_mask"]
        )
        
        # Pull logits - check if model returned a dict or a tensor
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        
        # Manually calculate MSE Loss (standard for regression)
        # Ensure labels are reshaped to match logits [Batch, 1] or [Batch]
        loss = F.mse_loss(logits.view(-1), sub_batch["labels"].view(-1))
        
        loss.backward()
        
        # Flatten LoRA gradients
        return torch.cat([p.grad.flatten() for n, p in model.named_parameters() 
                          if "lora_B" in n and p.grad is not None])

    iterator = iter(dataloader)
    
    for i in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
            
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Use our 'lang' tensor to split the batch
        en_mask = batch["lang"] == 0
        de_mask = batch["lang"] == 1
        
        if not en_mask.any() or not de_mask.any():
            continue 
            
        batch_en = {k: v[en_mask] for k, v in batch.items()}
        batch_de = {k: v[de_mask] for k, v in batch.items()}
        
        grad_en = get_grads(batch_en)
        grad_de = get_grads(batch_de)
        
        sim = F.cosine_similarity(grad_en, grad_de, dim=0).item()
        similarities.append(sim)
        print(f"Batch {i+1}/{num_batches} Similarity: {sim:.4f}")

    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        print(f"\nAverage Alignment (EN vs DE): {avg_sim:.4f}")
    
# Run it
#check_interference_sampled(model, dataset["train"], tokenizer)

def check_interference_balanced(model, dataset, tokenizer, num_batches=10, batch_size=16):
    model.train()
    
    # 1. Filter and Shuffle
    # Half the batch size per language to maintain total batch_size
    half_batch = batch_size // 2 
    
    ds_en = dataset.filter(lambda x: x['lang'] == 'eng_Latn').shuffle(seed=42)
    ds_de = dataset.filter(lambda x: x['lang'] == 'deu_Latn').shuffle(seed=42)
    
    similarities = []

    for i in range(num_batches):
        start = i * half_batch
        end = start + half_batch
        
        # 2. Balanced Selection
        # Ensure we don't exceed dataset limits
        if end > len(ds_en) or end > len(ds_de):
            print("Reached end of dataset.")
            break

        batch_en = ds_en.select(range(start, end))
        batch_de = ds_de.select(range(start, end))

        def get_grads(sample_ds):
            model.zero_grad()
            inputs = tokenizer(
                [str(text) for text in sample_ds["sentence"]], 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(model.device)
            
            # Ensure labels are long for classification or float for regression
            labels = torch.tensor(sample_ds["label"]).to(model.device)
            
            outputs = model(**inputs, labels=labels)
            outputs.loss.backward()
            
            return torch.cat([p.grad.flatten() for n, p in model.named_parameters() 
                              if "lora_B" in n and p.grad is not None])

        grad_en = get_grads(batch_en)
        grad_de = get_grads(batch_de)

        sim = F.cosine_similarity(grad_en, grad_de, dim=0).item()
        similarities.append(sim)
        print(f"Batch {i+1} Similarity: {sim:.4f}")

    avg_sim = sum(similarities) / len(similarities) if similarities else 0
    print(f"\nAverage Alignment (EN vs DE): {avg_sim:.4f}")
    return avg_sim