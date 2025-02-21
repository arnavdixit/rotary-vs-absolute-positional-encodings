from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from data import DialogSumDataset
import torch

def main():
    is_mps = torch.backends.mps.is_available()
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Load your dataset splits using your custom dataset class from data.py
    train_dataset = DialogSumDataset(split='train', tokenizer_name=model_name, max_length=256, summary_max_length=64)
    eval_dataset = DialogSumDataset(split='validation', tokenizer_name=model_name, max_length=256, summary_max_length=64)
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=4,   # Adjust according to your GPU memory
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=1000,
        logging_steps=100,
        fp16= not is_mps,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
if __name__ == '__main__':
    main()
    
    