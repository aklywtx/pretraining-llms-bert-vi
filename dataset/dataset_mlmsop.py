import os
import logging
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset

os.environ["HF_HOME"] = '/scratch/xtong/cache/'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/xtong/cache/'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)

class CreateDataset(Dataset):
    """
    Dataset wrapper for creating inputs suitable for Masked Language Modeling (MLM)
    and Sentence Order Prediction (SOP)/Next Sentence Prediction (NSP).
    """
    def __init__(self, dataset, tokenizer, max_length=256, mlm_probability=0.15):
        """
        Initializes dataset preparation for MLM and SOP tasks.

        Args:
            dataset (Dataset): HuggingFace dataset object.
            tokenizer (AutoTokenizer): Tokenizer object.
            max_length (int): Maximum sequence length.
            mlm_probability (float): Probability of masking tokens for MLM.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.vocab_size = len(tokenizer)

    def __len__(self):
        return len(self.dataset)
    
    def mask_sentence(self, sentence):
        """
        Prepares masked tokens inputs and labels for MLM task.

        Args:
            sentences (str): Input sentence.

        Returns:
            dict: A dictionary containing tensors for masked input IDs, attention masks, and labels.
        """
        encodings = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'][0]
        attention_mask = encodings['attention_mask'][0]
        
        labels = input_ids.clone()
        
        # Rest of the masking logic remains the same
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        random_words = torch.randint(0, self.vocab_size, labels.shape, dtype=torch.long)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        input_ids[indices_random] = random_words[indices_random]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        s1 = sample["sentence1"]
        s2 = sample["sentence2"]
        label = sample["label"]
        
        s1_dict = self.mask_sentence(s1)
        s2_dict = self.mask_sentence(s2)
        
        # Concatenate special tokens as tensors
        bos_tensor = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
        eos_tensor = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)

        s1 = torch.cat([bos_tensor, s1_dict["input_ids"], eos_tensor], dim=0)
        s2 = torch.cat([s2_dict["input_ids"], eos_tensor], dim=0)
        input_ids = torch.cat([s1, s2], dim=0)
        
        # Create attention mask
        s1_mask = torch.cat([torch.tensor([1], dtype=torch.long), s1_dict["attention_mask"], torch.tensor([1], dtype=torch.long)], dim=0)
        s2_mask = torch.cat([s2_dict["attention_mask"], torch.tensor([1], dtype=torch.long)], dim=0)
        attention_mask = torch.cat([s1_mask, s2_mask], dim=0)
        
        # Create MLM labels
        s1_label = torch.cat([bos_tensor, s1_dict["labels"], eos_tensor], dim=0)
        s2_label = torch.cat([s2_dict["labels"], eos_tensor], dim=0)
        mlm_labels = torch.cat([s1_label, s2_label], dim=0)

        # Create token type IDs (0 for s1, 1 for s2)
        token_type_ids = torch.cat(
            [torch.zeros(len(s1), dtype=torch.long), torch.ones(len(s2), dtype=torch.long)],
            dim=0
        )
        
        total_length = s1.size()[0] + s2.size()[0]
        if total_length < self.max_length:
            pad_length = self.max_length - total_length
            padding = torch.full((pad_length, ), self.tokenizer.pad_token_id, dtype=torch.long)
            padding_labels = torch.full((pad_length,), -100, dtype=torch.long)
            padding_mask = torch.zeros(pad_length, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=0)
            mlm_labels = torch.cat([mlm_labels, padding_labels], dim=0)
            token_type_ids = torch.cat([token_type_ids, padding_mask], dim=0)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=0)
        else:
            input_ids = input_ids[:self.max_length]
            mlm_labels = mlm_labels[:self.max_length]
            token_type_ids = token_type_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'mlm_labels': mlm_labels,
            'sop_label': label
        }


def create_dataloader(huggingface_path, batch_size=64, max_length=256):
    """
    Creates a DataLoader for training.

    Args:
        huggingface_path (str): Path or name of HuggingFace dataset.
        tokenizer (AutoTokenizer): Tokenizer instance.
        batch_size (int): Batch size for dataloader.
        max_length (int): Maximum sequence length.

    Returns:
        DataLoader: Configured DataLoader instance.
    """

    dataset = load_dataset(huggingface_path, split="train", cache_dir="/scratch/xtong/cache")
    tokenizer = AutoTokenizer.from_pretrained("vinai/PhoGPT-4B-Chat")
    tokenizer.add_special_tokens({'mask_token': '<MASK>'})
    # dataset = dataset.select(range(100000))

    # Create dataset
    mlmsop_dataset = CreateDataset(dataset, tokenizer, max_length=max_length)
    
    dataloader = DataLoader(
        mlmsop_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        prefetch_factor=2,
        pin_memory=True
    )
    
    return dataloader, tokenizer


# if __name__=="__main__":
#     dataloader, tokenizer = create_MLMSOP_dataloader(batch_size=64)
#     for batch in dataloader:
#         print(batch['input_ids'].shape)  # Should print (batch_size, max_length)
#         print(batch['attention_mask'].shape)