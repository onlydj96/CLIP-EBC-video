
# Original
# python3 test_nwpu.py --model clip-resnet50 --truncation 4 --weight_path weights/best_mae_0.pth --device cuda:0

# Torch custom
python3 test_torch.py --model clip-resnet50 --truncation 4 --weight_path weights/best_mae_0.pth --device cuda:0

# ONNX 
# python3 test_onnx.py --weight_path weights/clip_ebc.onnx --device cuda:0