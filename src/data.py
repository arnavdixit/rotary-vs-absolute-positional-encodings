import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class DialogSumDataset(Dataset):
    def __init__(self, split = 'train', tokenizer_name = 't5_small', max_length=256, summary_max_length=64):
        self.dataset = load_dataset("knkarthick/dialogsum")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.summary_max_length = summary_max_length
        self.split = split
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[self.split][idx]
        
        # Get the dialogue and summary fields from the sample
        dialogue = sample['dialogue']
        summary = sample['summary']
        
        # If dialogue is a list of utterances, join them into a single string
        if isinstance(dialogue, list):
            dialogue = " ".join(dialogue)
            
        # Tokenize the dialogue (input)
        dialogue_encodings = self.tokenizer(
            dialogue,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        
         # Tokenize the summary (target)
        summary_encodings = self.tokenizer(
            summary,
            truncation=True,
            padding='max_length',
            max_length=self.summary_max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": dialogue_encodings["input_ids"].squeeze(),        # Remove batch dimension
            "attention_mask": dialogue_encodings["attention_mask"].squeeze(),
            "labels": summary_encodings["input_ids"].squeeze()               # Use input_ids as labels
        }

if __name__ == '__main__':
    dataset = DialogSumDataset(split='train', tokenizer_name='t5-small', max_length=256, summary_max_length=64)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Print the shape of the first batch to verify dimensions
    for batch in dataloader:
        print("Input IDs shape:", batch['input_ids'].shape)         # Expected: (batch_size, max_length)
        print("Attention mask shape:", batch['attention_mask'].shape)   # Expected: (batch_size, max_length)
        print("Labels shape:", batch['labels'].shape)                 # Expected: (batch_size, summary_max_length)
        break