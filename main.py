import torch
import torch.nn as nn
import yaml
import json
import argparse
import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from src.model import Transformer
from src.trainer import Trainer
from src.tester import Tester
from dataset.dataset import get_dataloaders


def setup_logging(config):

    os.makedirs(config['train']['save_dir'], exist_ok=True)
    log_file = config['logging']['log_file']
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




def create_model(vocab_size, config, device):
    """Create model"""
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        num_layers=config['model']['num_layers'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        max_len=config['model']['max_len'],
        dropout=config['model']['dropout']
    ).to(device)
    
    return model


def train_mode(args, config, logger):

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        dataset_path=config['data']['dataset_path'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers']
    )
    
    vocab_size = len(vocab)
    
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    vocab_path = os.path.join(config['train']['save_dir'], 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    config_save_path = os.path.join(config['train']['save_dir'], 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    model = create_model(vocab_size, config, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        betas=tuple(config['train']['betas']),
        eps=config['train']['eps']
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=config['train']['label_smoothing']
    )
    
    total_steps = len(train_loader) * config['train']['num_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    tensorboard_dir = os.path.join(config['train']['save_dir'], 'tensorboard')
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['train']['num_epochs'],
        save_dir=config['train']['save_dir'],
        writer=writer
    )
    
    if args.resume:
        start_epoch, _ = trainer.load_checkpoint(args.resume)

    try:
        trainer.train()
    finally:
        writer.close()
    
    history_path = os.path.join(config['train']['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
        }, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")


def test_mode(args, config, logger):

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    vocab_path = os.path.join(config['train']['save_dir'], 'vocab.json')
    if not os.path.exists(vocab_path):
        logger.error(f"Error: Vocabulary file not found at {vocab_path}")
        logger.error("Please run training mode first to generate vocabulary")
        return
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    vocab_size = len(vocab)
    

    from dataset.dataset import CNNDailyMailDataset, collate_fn
    
    test_dataset = CNNDailyMailDataset(
        dataset_path=config['data']['dataset_path'],
        split='test',
        vocab=vocab,
        max_len=config['data']['src_max_len']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['test']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['train']['num_workers']
    )
 
    model = create_model(vocab_size, config, device)
    
    checkpoint_path = args.checkpoint or config['test']['checkpoint']
    if not os.path.exists(checkpoint_path):
        logger.error(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tester = Tester(
        model=model,
        test_loader=test_loader,
        vocab=vocab,
        device=device,
        config=config
    )
    
    num_samples = args.num_samples or config['test']['num_test_samples']
    results = tester.test(num_samples=num_samples)
    
    if args.evaluate:
        tester.evaluate_with_rouge()
    
    if args.save_results:
        results_path = os.path.join(config['train']['save_dir'], 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results[:100], f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Transformer Text Summarization - Train and Test')
    
    # Basic arguments
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help='Run mode: train or test')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file path')
    
    # Training arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint (train mode only)')
    
    # Testing arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint for testing (test mode only)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Perform ROUGE evaluation (test mode only)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save test results (test mode only)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to test (test mode only)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup logging
    logger = setup_logging(config)

    
    # Run based on mode
    if args.mode == 'train':
        train_mode(args, config, logger)
    elif args.mode == 'test':
        test_mode(args, config, logger)


if __name__ == '__main__':
    main()
