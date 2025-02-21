import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import evaluate
from torch.utils.data import DataLoader
from data import DialogSumDataset

import os
import json

def evaluate_model(model, tokenizer, dataset, device, batch_size=8):
        
    rouge = evaluate.load("rouge")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model.generate(
                input_ids,
                attention_mask = attention_mask,
                max_length = 64,
                num_beams = 4,
                early_stopping = True
            )
            
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens = True)
            decoded_refs = tokenizer.batch_decode(batch["labels"], skip_special_tokens = True)
            
            all_predictions.extend(decoded_preds)
            all_references.extend(decoded_refs)
            
    results = rouge.compute(predictions = all_predictions, references = all_references)
    
    return results

def save_rouge_scores(rouge_results, filename = 'rouge_scores.json'):
    # Create the results directory if it doesn't exist
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    file_path = os.path.join(results_dir, filename)
    with open(file_path, "w") as f:
        json.dump(rouge_results, f, indent=4)
        
    print(f"ROUGE scores saved to {file_path}")


def main():
    model_name = 'facebook/bart-base'
    
    model = BartForConditionalGeneration.from_pretrained("./results/checkpoint-3")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    test_dataset = DialogSumDataset(split = "test", tokenizer_name=model_name)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Evaluate the model
    rouge_results = evaluate_model(model, tokenizer, test_dataset, device=device, batch_size=8)
    save_rouge_scores(rouge_results, filename="baseline_rouge_scores.json")
    # Print the baseline ROUGE scores
    print("Baseline ROUGE scores:", rouge_results)
    
if __name__ == "__main__":
    main()
    