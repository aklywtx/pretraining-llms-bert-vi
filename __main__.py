from .dataset import create_dataloader
from .model import BERT
from .trainer import BERTTrainer
import argparse

parser = argparse.ArgumentParser(description="Arguments for training the BERT model.")

parser.add_argument("--language", choices=['vi'], type=str, help="Language for training data.")
parser.add_argument("--dataset", choices=['sop', 'nsp'], type=str, help="Dataset type: sentence order prediction (sop) or next sentence prediction (nsp).")
parser.add_argument("--max_length", choices=[128, 256, 384, 512], type=int, help="Maximum token length for input sequences.")
parser.add_argument("--embed_dim", choices=[384, 512, 768], type=int, help="Embedding dimension.")
parser.add_argument("--n_layers", type=int, help="Number of transformer layers.")
parser.add_argument("--num_heads", type=int, help="Number of attention heads.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer.")
parser.add_argument("--batch_size", choices=[16, 32, 64, 128, 256, 512, 1024], type=int, help="Batch size for training.")
parser.add_argument("--ff_dropout", type=float, default=0.0, help="Dropout rate for feed-forward layers.")
parser.add_argument("--id", type=int, help="ID for this training run")
parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
parser.add_argument(
    "--special",
    type=str,
    dest="special_condition",
    choices=["adam", "adamw_wd", "adam_lrd", "adam_wd_lrd", "adamw_wd_lrd"],
    help="Optimizer type: adam | adamw_wd | adam_lrd | adam_wd_lrd | adamw_wd_lrd (wd=weight decay, lrd=learning rate decay)."
)

args = parser.parse_args()

print(args)


def train():
    print("Creating Dataloader")   
    
    huggingface_path = f"WendyHoang/corpus_test_{args.dataset}"
    dataloader, tokenizer = create_dataloader(huggingface_path=huggingface_path, batch_size=args.batch_size, max_length=args.max_length)
    print("dataloader:", dataloader)
    
    
    print("Building BERT model")
    model_config = {
        'vocab_size': len(tokenizer),
        'embed_dim': args.embed_dim,
        'n_layers': args.n_layers,
        'num_heads': args.num_heads,
        'ff_dropout': args.ff_dropout
    }
    bert = BERT(**model_config)

    print("Creating BERT Trainer")
    NUM_EPOCHS=1
    trainer = BERTTrainer(bert, 
                          tokenizer=tokenizer,
                          dataloader=dataloader, 
                          optimizer=None, 
                          wandb_log=True, 
                          save_checkpoint_path=f'{args.output_dir}/BERT/{args.language}/output/model_checkpoints',
                          **model_config,
                          max_length=args.max_length,
                          language=args.language,
                          dataset=args.dataset,
                          epochs=NUM_EPOCHS,
                          lr=args.lr,
                          use_scheduler=False,
                          input_id=args.id,
                          batch_size=args.batch_size,
                          special_condition=args.special_condition
                          )
    print("Training Start")
    trainer.train()
    
if __name__=="__main__":
    train()
