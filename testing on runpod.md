# Testing on RunPod: End-to-End Execution Guide

This guide covers the necessary steps to deploy and run the modified `MLP testing` parameter-golf models over the cloud using an Nvidia H100 execution framework via RunPod.

### Step 1: Create a RunPod Account & Setup Credentials
1. Navigate to the [RunPod Deploy Console](https://console.runpod.io/deploy) and create an account.
2. Under the **Settings** menu, securely associate your local SSH public key. This is required so you can jump into the remote node seamlessly from your desktop.

### Step 2: Deploy the Parameter-Golf Pod
1. Parameter Golf requires heavy baseline CUDA libraries. A pre-configured template specifically designed for this repository is available here: [Parameter Golf Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th).
2. The template has Flash Attention and MLX GPU limits pre-installed. Leave options on default, but explicitly ensure **SSH terminal access** is toggled ON.
3. Deploy the Pod. Wait for the booting sequence to reach `Running` status, click **Connect** to get the SSH connection string, and execute it into your local terminal. You will automatically drop into the remote `/workspace/` directory.

### Step 3: Clone the Repository
Once inside your RunPod terminal, fetch your repository code:
```bash
cd /workspace
git clone https://github.com/AdityaP9116/parameter-golf.git
cd parameter-golf
```

### Step 4: Install Missing Dependencies & Download the FineWeb Dataset
Sometimes default templates miss the data-fetching packages. Inside the `parameter-golf` directory, install all required dependencies:
```bash
pip install numpy sentencepiece zstandard triton ninja huggingface_hub datasets tqdm flash-attn --no-build-isolation
```

Once installed, fetch the dataset caches and tokenizer payloads directly onto the remote container. Run the provided download script specifically passing the 1024-token vocabulary switch:
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```
*(Pro-Tip: If you are iterating on logic and just want a quick smoke-test instead of downloading all 80 data shards (8 Billion tokens), append `--train-shards 1` to skip heavy network latency)*.

### Step 5: Execute the ALBERT / Hash Script
After the dataset is mapped, you can trigger your distributed training pipeline via `torchrun`. 

For the **ALBERT** 14-Layer SwiGLU GQA configuration:
```bash
RUN_ID=albert_swiglu_14L_run \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 "MLP testing/train_gpt_I4_muon_cpp_axis_A_albert.py"
```

For the **Hash** 14-Layer SwiGLU GQA configuration:
```bash
RUN_ID=hash_swiglu_14L_run \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 "MLP testing/train_gpt_I4_muon_cpp_axis_A_hash.py"
```

### Optional Overrides
- **Bypass Time Limits**: The competition hard-caps training at roughly 10 minutes (600 seconds) of compute via exceptions. To test raw scalability without forced stops, append `MAX_WALLCLOCK_SECONDS=0` into the environment assignments.
- **Continuous Validation Logging**: If you want it to validate Bits-per-Byte continuously through the process rather than strictly at runtime completion, add `VAL_LOSS_EVERY=200` to the bash command.
