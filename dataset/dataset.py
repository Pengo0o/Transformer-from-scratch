import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from collections import Counter
import os


class CNNDailyMailDataset(Dataset):
    
    def __init__(self, dataset_path, split='train', vocab=None, max_len=512, max_vocab_size=50000):

        self.dataset_path = dataset_path
        self.split = split
        self.max_len = max_len
        self.max_vocab_size = max_vocab_size

        try:
            full_dataset = load_from_disk(dataset_path)
            self.data = full_dataset[split]
        except:
            split_path = os.path.join(dataset_path, split)
            self.data = load_from_disk(split_path)

        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
    
    def build_vocab(self):

        vocab = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        

        word_freq = Counter()

        sample_size = min(50000, len(self.data))
        
        for i in range(sample_size):
            article = self.data[i]['article']
            highlights = self.data[i]['highlights']

            article_tokens = self.simple_tokenize(article)
            highlights_tokens = self.simple_tokenize(highlights)
            
            word_freq.update(article_tokens)
            word_freq.update(highlights_tokens)
        
        most_common = word_freq.most_common(self.max_vocab_size - len(vocab))
        
        for word, _ in most_common:
            if word not in vocab:
                vocab[word] = len(vocab)
        
        return vocab
    
    def simple_tokenize(self, text):

        text = text.lower()

        for punct in '.,!?;:':
            text = text.replace(punct, f' {punct} ')
        tokens = text.split()
        return tokens
    
    def encode(self, text, max_len):

        tokens = self.simple_tokenize(text)
        
        if len(tokens) > max_len - 2:  
            tokens = tokens[:max_len - 2]
        
        indices = [self.vocab['<sos>']]
        for token in tokens:
            indices.append(self.vocab.get(token, self.vocab['<unk>']))
        indices.append(self.vocab['<eos>'])
        
        return indices
    
    def __len__(self):
        return 15
        # return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        article = item['article']
        highlights = item['highlights']
        

        src_indices = self.encode(article, max_len=512)  
        tgt_indices = self.encode(highlights, max_len=150)  
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)


def collate_fn(batch):

    src_batch, tgt_batch = zip(*batch)
    
    src_max_len = max([len(s) for s in src_batch])
    tgt_max_len = max([len(t) for t in tgt_batch])
    
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):

        src_pad_len = src_max_len - len(src)
        tgt_pad_len = tgt_max_len - len(tgt)
        
        src_padded.append(torch.cat([src, torch.zeros(src_pad_len, dtype=torch.long)]))
        tgt_padded.append(torch.cat([tgt, torch.zeros(tgt_pad_len, dtype=torch.long)]))
    
    return torch.stack(src_padded), torch.stack(tgt_padded)


def get_dataloaders(dataset_path, batch_size=8, num_workers=0):

    train_dataset = CNNDailyMailDataset(
        dataset_path=dataset_path,
        split='train',
        vocab=None,  
        max_len=512
    )
    
    val_dataset = CNNDailyMailDataset(
        dataset_path=dataset_path,
        split='validation',
        vocab=train_dataset.vocab,
        max_len=512
    )
    
    test_dataset = CNNDailyMailDataset(
        dataset_path=dataset_path,
        split='test',
        vocab=train_dataset.vocab,
        max_len=512
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, train_dataset.vocab


