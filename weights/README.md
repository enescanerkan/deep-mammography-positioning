# Pretrained weights

Weights are not tracked by git (`*.pth` is ignored). Place them here or point the
corresponding environment variable at them.

## RadImageNet ResNet-50 (`RadImageNet_ResNet50_alt.pth`)

Used by the dual-stream classifier backbone `resnet50_radimagenet`.

| | |
|---|---|
| Source | https://huggingface.co/convergedmachine/RadImagenet (file `resnet50.pth`, MIT license) |
| Size | 191,145,887 bytes |
| SHA-256 | `b6daa90de92589c404e9da30f4c0ce49182763d4a952f1222caeec647237bacf` |
| Format | `{'model': state_dict}` with a torch.compile `_orig_mod.` prefix and a 165-way `fc`; both are stripped on load |

```bash
# from the repository root
mkdir -p weights
curl -L -o weights/RadImageNet_ResNet50_alt.pth \
  https://huggingface.co/convergedmachine/RadImagenet/resolve/main/resnet50.pth
sha256sum weights/RadImageNet_ResNet50_alt.pth
```

To keep the file elsewhere:

```bash
export RADIMAGENET_RESNET50_PATH=/path/to/resnet50.pth
```

The loader is `_load_radimagenet_resnet50()` in
`dual-stream-classification/utils/dual_models.py`.
