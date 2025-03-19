import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from ..model import BERT
from ..model import BERTLM

# Load the checkpoint
model_name = "NSP_8layers"
checkpoint_path = "checkpoint_final.pth"
checkpoint = torch.load(checkpoint_path)

# Define max sequence length
maxlen = 256
print(f"Max sequence length inferred from filename: {maxlen}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/PhoGPT-4B-Chat")
tokenizer.add_special_tokens({'mask_token': '<MASK>'})
vocab_size = len(tokenizer)
data = pd.read_csv('evaluation_data/faithfulness.csv')

# Evaluation Dataset Class
class EvaluationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = row['Original']
        option1 = row['Correct']
        option2 = row['Incorrect']

        def tokenize_text(text):
            tokens = self.tokenizer(
                text, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors='pt'
            )
            return torch.cat([
                torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long),
                tokens["input_ids"].squeeze(),
                torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            ])

        original = tokenize_text(sentence)
        s1 = tokenize_text(option1)
        s2 = tokenize_text(option2)

        def pad_sequence(seq):
            pad_length = self.max_length - seq.size(0)
            if pad_length > 0:
                padding = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                return torch.cat([seq, padding])
            return seq[:self.max_length]

        original, s1, s2 = map(pad_sequence, [original, s1, s2])
        
        original_mask = (original != self.tokenizer.pad_token_id).long()
        s1_mask = (s1 != self.tokenizer.pad_token_id).long()
        s2_mask = (s2 != self.tokenizer.pad_token_id).long()

        return {
            'original': original, 's1': s1, 's2': s2,
            's1_attention_mask': s1_mask, 's2_attention_mask': s2_mask, 'original_attention_mask': original_mask,
            's1_token_type_ids': torch.zeros_like(s1), 's2_token_type_ids': torch.ones_like(s2),
            'original_token_type_ids': torch.zeros_like(original), 'original_row': row.to_dict()
        }

# Load Evaluation Data
eval_dataset = EvaluationDataset(data, tokenizer, max_length=maxlen)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Load Model
model = BERTLM(BERT(
    vocab_size=checkpoint['vocab_size'],
    embed_dim=checkpoint['embed_dim'],
    n_layers=checkpoint['n_layers'],
    num_heads=checkpoint['num_heads']
), vocab_size)

model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Evaluation
correct_predictions, predictions = 0, []

for batch in eval_dataloader:
    with torch.no_grad():
        s1, _, _ = model(batch['s1'], batch['s1_token_type_ids'], batch['s1_attention_mask'].unsqueeze(1))
        s2, _, _ = model(batch['s2'], batch['s2_token_type_ids'], batch['s2_attention_mask'].unsqueeze(1))
        o, _, _ = model(batch['original'], batch['original_token_type_ids'], batch['original_attention_mask'].unsqueeze(1))
    
    for i in range(len(batch['original'])):
        is_correct = torch.nn.functional.cosine_similarity(o[:, 0, :], s1[:, 0, :]) > \
                     torch.nn.functional.cosine_similarity(o[:, 0, :], s2[:, 0, :])
        correct_predictions += is_correct.item()
        
        predictions.append({
            'Original': batch['original_row']['Original'][i],
            'Sentence1': batch['original_row']['Correct'][i],
            'Sentence2': batch['original_row']['Incorrect'][i],
            'Prediction': 'Correct' if is_correct else 'Incorrect'
        })

# Accuracy Calculation
accuracy = correct_predictions / len(data)
checkpoint_name = f"faithfulness_{model_name}"

# Save Results
with open(f'result/{checkpoint_name}_accuracy.txt', "w") as outfile:
    outfile.write(f"Accuracy: {accuracy * 100:.2f}%\n")
print(f"Accuracy: {accuracy * 100:.2f}%")

output_df = pd.DataFrame(predictions)
output_df.to_csv(f'result/{checkpoint_name}_output.csv', index=False)
print(f"Predictions saved to {checkpoint_name}")
