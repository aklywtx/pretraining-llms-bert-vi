import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from ..model import BERT
from ..model import BERTLM
import re

# Load checkpoint
model_name = "NSP_8layers"
checkpoint_path = "checkpoint_final.pth"
checkpoint = torch.load(checkpoint_path)

# Set max sequence length
maxlen = 256
print(f"Max sequence length inferred from filename: {maxlen}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/PhoGPT-4B-Chat")
tokenizer.add_special_tokens({'mask_token': '<MASK>'})
vocab_size = len(tokenizer)
data = pd.read_csv('evaluation_data/Winograd.csv')

# Evaluation dataset class
class EvaluationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        s1, s2 = row['Sentence1'], row['Sentence2']
        s2 = s2.replace('<mask>', self.tokenizer.mask_token).replace('<MASK>', self.tokenizer.mask_token)
        correct_token, incorrect_token = row['Correct'], row['Incorrect']

        # Standardize token replacements
        s1 = s1.replace(correct_token, 'Nam').replace(incorrect_token, 'Minh')
        s2 = s2.replace(correct_token, 'Nam').replace(incorrect_token, 'Minh')
        correct_token, incorrect_token = 'Nam', 'Minh'
        
        # Tokenize sentences
        s1_dict = self.tokenizer(s1, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors='pt')
        s2_dict = self.tokenizer(s2, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors='pt')

        # Add special tokens
        bos_tensor = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
        eos_tensor = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        s1 = torch.cat([bos_tensor, s1_dict["input_ids"].squeeze(0), eos_tensor])
        s2 = torch.cat([s2_dict["input_ids"].squeeze(0), eos_tensor])
        input_ids = torch.cat([s1, s2])
        
        # Attention and token type masks
        s1_mask = torch.cat([torch.tensor([1]), s1_dict["attention_mask"].squeeze(0), torch.tensor([1])])
        s2_mask = torch.cat([s2_dict["attention_mask"].squeeze(0), torch.tensor([1])])
        attention_mask = torch.cat([s1_mask, s2_mask])
        token_type_ids = torch.cat([torch.zeros(len(s1)), torch.ones(len(s2))]).long()
        
        # Padding
        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            padding = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            token_type_ids = torch.cat([token_type_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        else:
            input_ids, token_type_ids, attention_mask = input_ids[:self.max_length], token_type_ids[:self.max_length], attention_mask[:self.max_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'correct_token': correct_token,
            'incorrect_token': incorrect_token,
            'original_row': row.to_dict()
        }

# Load evaluation data
eval_dataset = EvaluationDataset(data, tokenizer, max_length=maxlen)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Load model
model = BERTLM(BERT(vocab_size=checkpoint['vocab_size'],
                    embed_dim=checkpoint['embed_dim'],
                    n_layers=checkpoint['n_layers'],
                    num_heads=checkpoint['num_heads']), vocab_size)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Evaluation loop
correct_predictions = 0
predictions = []

for batch in eval_dataloader:
    input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
    with torch.no_grad():
        _, mlm_logits, _ = model(input_ids, token_type_ids, attention_mask.unsqueeze(1))
    
    for i in range(len(batch['correct_token'])):
        mask_index = (input_ids[i] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        mask_logits = mlm_logits[i, mask_index, :]
        
        correct_id = tokenizer.convert_tokens_to_ids(batch['correct_token'][i])
        incorrect_id = tokenizer.convert_tokens_to_ids(batch['incorrect_token'][i])
        
        correct_prob = torch.softmax(mask_logits, dim=-1)[0, correct_id].item()
        incorrect_prob = torch.softmax(mask_logits, dim=-1)[0, incorrect_id].item()
        
        is_correct = correct_prob > incorrect_prob
        correct_predictions += is_correct
        
        predictions.append({
            'Sentence1': batch['original_row']['Sentence1'][i],
            'Sentence2': batch['original_row']['Sentence2'][i],
            'Correct_Token': batch['correct_token'][i],
            'Incorrect_Token': batch['incorrect_token'][i],
            'Correct_Probability': correct_prob,
            'Incorrect_Probability': incorrect_prob,
            'Prediction': 'Correct' if is_correct else 'Incorrect'
        })

# Compute accuracy
accuracy = correct_predictions / len(data)
checkpoint_name = f"winograd_{model_name}"

# Save results
accuracy_path = f"result/{checkpoint_name}_accuracy.txt"
output_path = f"result/{checkpoint_name}_output.csv"

with open(accuracy_path, "w") as outfile:
    outfile.write(f"Accuracy: {accuracy * 100:.2f}%\n")
print(f"Accuracy: {accuracy * 100:.2f}%")

pd.DataFrame(predictions).to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
