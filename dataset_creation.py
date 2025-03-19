from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from underthesea import sent_tokenize
import re
import random
from huggingface_hub import login
import pandas as pd

ds = load_dataset("wikimedia/wikipedia", "20231101.vi")
ds_news = load_dataset("bkai-foundation-models/BKAINewsCorpus", split="train")
train_dataset = ds['train']
login(token="huggingface_token")
name_corpus = "WendyHoang/corpus_test_nsp"
char_cap_wiki = 1000
char_cap_news = 1000
type_data = 'NSP'


def count_non_whitespace_chars(article_text):
    """
    Count non-whitespace characters in the article text after removing HTML tags.
    """
    non_whitespace_chars = len(re.findall(r'\S', article_text))
    return non_whitespace_chars


def get_sampled_dataset(dataset, char_cap, text='text', seed=42):
    """
    Samples from the dataset up to a character cap. Ensures no document is truncated,
    and the total can slightly exceed the cap if adding one more document exceeds the limit.
    """
    dataset = dataset.shuffle(seed=seed)

    total_chars = 0
    sampled_data = []

    for example in dataset:
        # Ensure 'example' is a dictionary and contains 'text'
        if isinstance(example, dict) and text in example:
            doc_length = count_non_whitespace_chars(example[text])
            sampled_data.append(example)
            total_chars += doc_length
            if total_chars + doc_length > char_cap:
                break

    print(total_chars)
    return Dataset.from_list(sampled_data)


data = get_sampled_dataset(train_dataset, char_cap_wiki)
data_news = get_sampled_dataset(ds_news, char_cap_news)


def tokenize_dataset(example):
    sentences = sent_tokenize(example['text'])
    # Filter out very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
    return {'sentences': sentences}


processed_dataset = data.map(tokenize_dataset, remove_columns=['text'])
processed_news = data_news.map(tokenize_dataset, remove_columns=['text'])
print(processed_dataset[0])
print(processed_news[0])

# Collect all sentences from the corpus to sample random negative pairs
all_sentences_wiki = []
all_sentences_news = []
if type_data == 'NSP':
    for example in processed_dataset:
        all_sentences_wiki.extend(example['sentences'])

    for example in processed_news:
        all_sentences_news.extend(example['sentences'])


def generate_dataset_examples(example, corpus_sentences, type_data='SOP'):
    sentences = example['sentences']
    num_sentences = len(sentences)

    sentence1_list = []
    sentence2_list = []
    labels = []

    for i in range(num_sentences - 1):
        # Positive pair (original order)
        sentence1_list.append(sentences[i])
        sentence2_list.append(sentences[i + 1])
        labels.append(1)

        # Negative pair (random sentence from the corpus, not the next sentence)
        if type_data == 'SOP':
            sentence1_list.append(sentences[i + 1])
            sentence2_list.append(sentences[i])
        else:
            random_sentence = random.choice(corpus_sentences)
            while random_sentence == sentences[i + 1]:
                random_sentence = random.choice(corpus_sentences)
            sentence1_list.append(sentences[i])
            sentence2_list.append(random_sentence)
        labels.append(0)

    return {'sentence1': sentence1_list, 'sentence2': sentence2_list, 'label': labels}


final_dataset = []
if type_data == 'SOP':
    sop_dataset = processed_dataset.map(
        lambda example: generate_dataset_examples(example, all_sentences_wiki),
        batched=False, remove_columns=['sentences', 'id', 'url', 'title'], num_proc=8
    )

    sop_new = processed_news.map(
        lambda example: generate_dataset_examples(example, all_sentences_news),
        batched=False, remove_columns=['sentences', 'id', 'link', 'publish'], num_proc=8
    )
    final_dataset = concatenate_datasets([sop_dataset, sop_new])
else:
    nsp_dataset = processed_dataset.map(
        lambda example: generate_dataset_examples(example, all_sentences_wiki, type_data='NSP'),
        batched=False, remove_columns=['sentences', 'id', 'url', 'title'], num_proc=8
    )

    nsp_new = processed_news.map(
        lambda example: generate_dataset_examples(example, all_sentences_news, type_data='NSP'),
        batched=False, remove_columns=['sentences', 'id', 'link', 'publish'], num_proc=8
    )
    final_dataset = concatenate_datasets([nsp_dataset, nsp_new])

# flatten data
sentence1_list = []
sentence2_list = []
labels = []
for row in final_dataset:
    sentence1_list += row['sentence1']
    sentence2_list += row['sentence2']
    labels += row['label']

data_list = pd.DataFrame(
    {'sentence1': sentence1_list,
     'sentence2': sentence2_list,
     'label': labels
     })

data = DatasetDict({'train': Dataset.from_pandas(data_list)})
print(data)
data.push_to_hub(name_corpus)