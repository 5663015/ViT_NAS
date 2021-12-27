- ```./train_vit```：ViT在ImageNet上的训练

```
export CUDA_VISIBLE_DEVICES=0,1 && nohup python -u train.py --arch vit_s --checkpoint ./checkpoints > vit_s.log 2>&1 &
```

- ```./AutoFormer```：Transformer NAS，参考自[Cream/AutoFormer at main · microsoft/Cream (github.com)](https://github.com/microsoft/Cream/tree/main/AutoFormer)

