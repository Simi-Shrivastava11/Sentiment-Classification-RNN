import subprocess

# Simple script to test different combinations to see what works best
experiments = [
    # Testing different architectures
    "python src/train.py --arch rnn --activation relu --optimizer adam --seq_len 50 --epochs 5",
    "python src/train.py --arch lstm --activation relu --optimizer adam --seq_len 50 --epochs 5",
    "python src/train.py --arch bi_lstm --activation relu --optimizer adam --seq_len 50 --epochs 5",
    
    # Testing different activation functions 
    "python src/train.py --arch lstm --activation tanh --optimizer adam --seq_len 50 --epochs 5",
    "python src/train.py --arch lstm --activation sigmoid --optimizer adam --seq_len 50 --epochs 5",
    
    # Testing different optimizers 
    "python src/train.py --arch lstm --activation relu --optimizer sgd --seq_len 50 --epochs 5",
    "python src/train.py --arch lstm --activation relu --optimizer rmsprop --seq_len 50 --epochs 5",
    
    # Testing different sequence lengths 
    "python src/train.py --arch lstm --activation relu --optimizer adam --seq_len 25 --epochs 5",
    "python src/train.py --arch lstm --activation relu --optimizer adam --seq_len 100 --epochs 5",
    
    # Testing gradient clipping 
    "python src/train.py --arch rnn --activation relu --optimizer adam --seq_len 50 --epochs 5 --clip",
    "python src/train.py --arch lstm --activation relu --optimizer adam --seq_len 50 --epochs 5 --clip",
    "python src/train.py --arch bi_lstm --activation relu --optimizer adam --seq_len 50 --epochs 5 --clip",
    
    # Testing bi_lstm with different settings
    "python src/train.py --arch bi_lstm --activation relu --optimizer adam --seq_len 25 --epochs 5",
    "python src/train.py --arch bi_lstm --activation relu --optimizer adam --seq_len 100 --epochs 5",
    "python src/train.py --arch bi_lstm --activation relu --optimizer sgd --seq_len 25 --epochs 5",
    "python src/train.py --arch bi_lstm --activation relu --optimizer sgd --seq_len 50 --epochs 5",
    "python src/train.py --arch bi_lstm --activation relu --optimizer sgd --seq_len 100 --epochs 5",
    "python src/train.py --arch bi_lstm --activation relu --optimizer rmsprop --seq_len 25 --epochs 5",
    "python src/train.py --arch bi_lstm --activation relu --optimizer rmsprop --seq_len 50 --epochs 5",
    "python src/train.py --arch bi_lstm --activation relu --optimizer rmsprop --seq_len 100 --epochs 5",
    "python src/train.py --arch bi_lstm --activation tanh --optimizer adam --seq_len 50 --epochs 5",
    "python src/train.py --arch bi_lstm --activation sigmoid --optimizer adam --seq_len 50 --epochs 5",
    
    # Testing lstm with more combinations 
    "python src/train.py --arch lstm --activation tanh --optimizer adam --seq_len 25 --epochs 5",
    "python src/train.py --arch lstm --activation tanh --optimizer adam --seq_len 100 --epochs 5",
    "python src/train.py --arch lstm --activation sigmoid --optimizer adam --seq_len 25 --epochs 5",
    "python src/train.py --arch lstm --activation sigmoid --optimizer adam --seq_len 100 --epochs 5",
    "python src/train.py --arch lstm --activation relu --optimizer sgd --seq_len 25 --epochs 5",
    "python src/train.py --arch lstm --activation relu --optimizer sgd --seq_len 100 --epochs 5",
    "python src/train.py --arch lstm --activation relu --optimizer rmsprop --seq_len 25 --epochs 5",
    "python src/train.py --arch lstm --activation relu --optimizer rmsprop --seq_len 100 --epochs 5",
    
    # Testing rnn with more settings 
    "python src/train.py --arch rnn --activation relu --optimizer adam --seq_len 25 --epochs 5",
    "python src/train.py --arch rnn --activation relu --optimizer adam --seq_len 100 --epochs 5",
    "python src/train.py --arch rnn --activation tanh --optimizer adam --seq_len 50 --epochs 5",
    "python src/train.py --arch rnn --activation sigmoid --optimizer adam --seq_len 50 --epochs 5",
    "python src/train.py --arch rnn --activation relu --optimizer sgd --seq_len 50 --epochs 5",
    "python src/train.py --arch rnn --activation relu --optimizer rmsprop --seq_len 50 --epochs 5",
]

# Running each experiment
for i, command in enumerate(experiments):
    print(f"Experiment {i+1}/{len(experiments)}")
    print(f"Running: {command}")
    
    subprocess.run(command.split())
    print(f"Finished experiment {i+1}")
