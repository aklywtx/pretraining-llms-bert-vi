import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from ..model.bert import BERT
from ..model.bertlm import BERTLM


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
data = pd.read_csv('evaluation_data/repron_vn_binary.csv')

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
        sentence = row['Sentence'].replace('<mask>', self.tokenizer.mask_token).replace('<MASK>', self.tokenizer.mask_token)
        correct_token = row['Correct']
        incorrect_token = row['Incorrect']

        correct_token_ids = self.tokenizer.encode(correct_token, add_special_tokens=False)
        incorrect_token_ids = self.tokenizer.encode(incorrect_token, add_special_tokens=False)

        s1 = sentence.replace(self.tokenizer.mask_token, ''.join([self.tokenizer.mask_token] * len(correct_token_ids)))
        s2 = sentence.replace(self.tokenizer.mask_token, ''.join([self.tokenizer.mask_token] * len(incorrect_token_ids)))

        def tokenize_text(text):
            tokens = self.tokenizer(
                text, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors='pt'
            )
            return torch.cat([
                torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long),
                tokens["input_ids"].squeeze(),
                torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            ])

        input_ids_1, input_ids_2 = tokenize_text(s1), tokenize_text(s2)

        def pad_sequence(seq):
            pad_length = self.max_length - seq.size(0)
            if pad_length > 0:
                padding = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                return torch.cat([seq, padding])
            return seq[:self.max_length]

        input_ids_1, input_ids_2 = map(pad_sequence, [input_ids_1, input_ids_2])
        
        attention_mask_1 = (input_ids_1 != self.tokenizer.pad_token_id).long()
        attention_mask_2 = (input_ids_2 != self.tokenizer.pad_token_id).long()
        
        token_type_ids_1 = torch.zeros_like(input_ids_1)
        token_type_ids_2 = torch.ones_like(input_ids_2)

        return {
            'sentence': sentence,
            'correct': correct_token,
            'incorrect': incorrect_token,
            'input_ids_1': input_ids_1,
            'attention_mask_1': attention_mask_1,
            'token_type_ids_1': token_type_ids_1,
            'input_ids_2': input_ids_2,
            'attention_mask_2': attention_mask_2,
            'token_type_ids_2': token_type_ids_2,
            'correct_token_ids': correct_token_ids,
            'incorrect_token_ids': incorrect_token_ids,
            'original_row': row.to_dict()
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
        _, mlm_logits_1, _ = model(batch['input_ids_1'], batch['token_type_ids_1'], batch['attention_mask_1'].unsqueeze(1))
        _, mlm_logits_2, _ = model(batch['input_ids_2'], batch['token_type_ids_2'], batch['attention_mask_2'].unsqueeze(1))

    sub_token_precisions_1 = [torch.log_softmax(mlm_logits_1[0, idx, :], dim=-1)[batch['correct_token_ids'][idx]].item() for idx in (batch['input_ids_1'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]]
    sub_token_precisions_2 = [torch.log_softmax(mlm_logits_2[0, idx, :], dim=-1)[batch['incorrect_token_ids'][idx]].item() for idx in (batch['input_ids_2'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]]

    score_1, score_2 = sum(sub_token_precisions_1) / len(sub_token_precisions_1), sum(sub_token_precisions_2) / len(sub_token_precisions_2)
    is_correct = score_1 > score_2
    correct_predictions += is_correct

    predictions.append({
        'sentence': batch['sentence'],
        'Correct': batch['correct'],
        'Incorrect': batch['incorrect'],
        'Correct_Probability': score_1,
        'Incorrect_Probability': score_2,
        'Prediction': 'Correct' if is_correct else 'Incorrect'
    })

# Accuracy Calculation
accuracy = correct_predictions / len(data)
checkpoint_name = f"realworld_{model_name}"

# Save Results
with open(f'result/{checkpoint_name}_accuracy.txt', "w") as outfile:
    outfile.write(f"Accuracy: {accuracy * 100:.2f}%\n")
print(f"Accuracy: {accuracy * 100:.2f}%")

output_df = pd.DataFrame(predictions)
output_df.to_csv(f'result/{checkpoint_name}_output.csv', index=False)
print(f"Predictions saved to {checkpoint_name}")