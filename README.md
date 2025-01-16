# Setup
```
pip install -e .
```

## Need to install EasyVideo
```
git clone https://github.com/OurBluePrint/easy_video
cd easy_video
pip install -e .
```

# Training Stage 1
```
python scripts/train.py --config configs/config_default.yaml # 1 GPU
python scripts/train_multiGPU.py --config configs/config_default.yaml # multi-GPU
```

