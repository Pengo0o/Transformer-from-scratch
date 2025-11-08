import torch
from tqdm import tqdm
import os


class Tester:

    def __init__(self, model, test_loader, vocab, device, config):
        
        self.model = model
        self.test_loader = test_loader
        self.vocab = vocab
        self.device = device
        self.config = config

        self.idx_to_word = {idx: word for word, idx in vocab.items()}
        
        self.model.eval()
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx == self.vocab['<eos>']:
                break
            if idx not in [self.vocab['<pad>'], self.vocab['<sos>']]:
                words.append(self.idx_to_word.get(idx, '<unk>'))
        return ' '.join(words)
    
    def generate_greedy(self, src, max_len):
        with torch.no_grad():
            tgt = torch.tensor([[self.vocab['<sos>']]], dtype=torch.long).to(self.device)
            
            for _ in range(max_len):
                output = self.model(src, tgt)
                next_token = output[0, -1, :].argmax().item()
                
                if next_token == self.vocab['<eos>']:
                    break
                
                tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(self.device)], dim=1)
            
            return tgt[0].cpu().tolist()
    
    def generate_beam_search(self, src, beam_width, max_len):
        with torch.no_grad():
            beams = [([self.vocab['<sos>']], 0.0)]
            
            for _ in range(max_len):
                candidates = []
                
                for seq, score in beams:
                    if seq[-1] == self.vocab['<eos>']:
                        candidates.append((seq, score))
                        continue
                    
                    tgt = torch.tensor([seq], dtype=torch.long).to(self.device)
                    output = self.model(src, tgt)
                    
                    next_token_logits = output[0, -1, :]
                    log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    
                    top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
                    
                    for prob, idx in zip(top_k_probs, top_k_indices):
                        new_seq = seq + [idx.item()]
                        new_score = score + prob.item()
                        candidates.append((new_seq, new_score))
                
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]
                
                if all(seq[-1] == self.vocab['<eos>'] for seq, _ in beams):
                    break
            
            best_seq, _ = beams[0]
            return best_seq
    
    def generate(self, src):
        method = self.config['test']['decode_method']
        max_len = self.config['test']['max_generate_len']
        
        if method == 'greedy':
            return self.generate_greedy(src, max_len)
        elif method == 'beam_search':
            beam_width = self.config['test']['beam_width']
            return self.generate_beam_search(src, beam_width, max_len)

    def test(self, num_samples=None, show_examples=True):

        if num_samples is None:
            num_samples = len(self.test_loader.dataset)

        results = []
        sample_count = 0

        for batch_idx, (src, tgt) in enumerate(tqdm(self.test_loader, desc="Testing")):
            src = src.to(self.device)
            
            for i in range(src.size(0)):
                if sample_count >= num_samples:
                    break
                
                src_input = src[i:i+1]
                generated_indices = self.generate(src_input)
                generated_summary = self.decode(generated_indices)
                
                reference_summary = self.decode(tgt[i])
                
                article = self.decode(src[i])
                
                results.append({
                    'article': article,
                    'reference': reference_summary,
                    'generated': generated_summary
                })
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break
        
        return results
    
    def evaluate_with_rouge(self):

        from rouge_score import rouge_scorer

        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
        }
        
        num_samples = self.config['evaluate']['num_samples']
        sample_count = 0
        
        for src, tgt in tqdm(self.test_loader, desc="评估中"):
            src = src.to(self.device)
            
            for i in range(src.size(0)):
                if sample_count >= num_samples:
                    break
 
                src_input = src[i:i+1]
                generated_indices = self.generate(src_input)
                generated_summary = self.decode(generated_indices)

                reference_summary = self.decode(tgt[i])

                scores = scorer.score(reference_summary, generated_summary)
                
                for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                    rouge_scores[rouge_type]['precision'].append(scores[rouge_type].precision)
                    rouge_scores[rouge_type]['recall'].append(scores[rouge_type].recall)
                    rouge_scores[rouge_type]['fmeasure'].append(scores[rouge_type].fmeasure)
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break

        avg_scores = {}
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            avg_scores[rouge_type] = {
                'precision': sum(rouge_scores[rouge_type]['precision']) / sample_count,
                'recall': sum(rouge_scores[rouge_type]['recall']) / sample_count,
                'fmeasure': sum(rouge_scores[rouge_type]['fmeasure']) / sample_count
            }

        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            print(f"\n{rouge_type.upper()}:")
            print(f"  Precision: {avg_scores[rouge_type]['precision']:.4f}")
            print(f"  Recall:    {avg_scores[rouge_type]['recall']:.4f}")
            print(f"  F-measure: {avg_scores[rouge_type]['fmeasure']:.4f}")
        
        return avg_scores

