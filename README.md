# COMET-PPO

Train [comet](https://github.com/allenai/comet-atomic-2020) (allenai, 2021) with PPO.

## Download COMET weights

Download and unzip comet weights from the official [repo](https://github.com/allenai/comet-atomic-2020) and put them in
`comet_pretrained/` directory.

```bash
wget https://storage.googleapis.com/ai2-mosaic-public/projects/mosaic-kgs/comet-atomic_2020_BART.zip
wget https://storage.googleapis.com/ai2-mosaic-public/projects/mosaic-kgs/comet-atomic_2020_GPT2XL.zip

unzip comet-atomic_2020_BART.zip
unzip comet-atomic_2020_GPT2XL.zip

mkdir comet_pretrained/

mv comet-atomic_2020_BART comet_pretrained/comet-atomic_2020_BART
mv gpt2xl-comet-atomic-2020 comet_pretrained/gpt2xl-comet-atomic-2020
```


* For GPT2 weights, you need to move the tokenizer files in `tokenizer/` directory
  out to the parent directory that contains the model weights.

```bash
cd comet_pretrained/gpt2xl-comet-atomic-2020
mv tokenizer/* ./
rmdir tokenizer
```
## Setup

The code uses `trl` along with `transformers`.

```bash
pip install -r requirements.txt
```

### Train Larger Models on Small GPUs

* You may want to uncomment larger model related packages in requirements.txt. Also,
  `peft` might come in handy.

```bash
git clone https://github.com/huggingface/peft.git
cd peft
pip install .

pip install bitsandbytes loralib
```

* Although deepspeed integration is out of scope for this repo, you may try deepspeed,
  as `trl` supports deepspeed.

## Run experiments

1. First, login to [wandb](wandb.ai). `wandb login`.

```bash
sh scripts/train-gpt.sh 0,1,2,3
```

Training bash scripts take `gpu ids` (0,1,2,3 above) as argument. If you use just one GPU,
it will run using a simple python launcher (`python main.py`). If you use multi-gpu,
it will automatically use `torchrun` to launch.

* multi-node training is not supported yet.

