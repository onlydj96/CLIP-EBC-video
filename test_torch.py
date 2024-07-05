import torch
from argparse import ArgumentParser
import os, json
from tqdm import tqdm
import time

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import CustomDataset
from models import get_model
from utils import get_config, extract_coordinates, save_prediction_image_with_dot, save_prediction_image_with_heatmap

parser = ArgumentParser(description="Test a trained model on the Custom test set.")
# Parameters for model
parser.add_argument("--model", type=str, default="resnet50", help="The model to train.")
parser.add_argument("--input_size", type=int, default=448, help="The size of the input image.")
parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32], help="The reduction factor of the model.")
parser.add_argument("--regression", action="store_true", help="Use blockwise regression instead of classification.")
parser.add_argument("--truncation", type=int, default=None, help="The truncation of the count.")
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"], help="The representative count values of bins.")
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"], help="The prompt type for CLIP.")
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"], help="The granularity of bins.")
parser.add_argument("--num_vpt", type=int, default=32, help="The number of visual prompt tokens.")
parser.add_argument("--vpt_drop", type=float, default=0.0, help="The dropout rate for visual prompt tokens.")
parser.add_argument("--shallow_vpt", action="store_true", help="Use shallow visual prompt tokens.")
parser.add_argument("--weight_path", type=str, default="weights/best_mae_0.pth", help="The path to the weights of the model.")

parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--device", type=str, default="cuda", help="The device to use for evaluation.")
parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for the data loader.")



    
def main(args):

    print("Testing a trained model on the NWPU-Crowd test set.")
    device = torch.device(args.device)
    _ = get_config(vars(args).copy(), mute=False)
    
    with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
        config = json.load(f)[str(args.truncation)]["nwpu"]
    bins = config["bins"][args.granularity]
    anchor_points = config["anchor_points"][args.granularity]["average"] if args.anchor_points == "average" else config["anchor_points"]["middle"]
    bins = [(float(b[0]), float(b[1])) for b in bins]
    anchor_points = [float(p) for p in anchor_points]

    args.bins = bins
    args.anchor_points = anchor_points
    torch.cuda.reset_peak_memory_stats()

    model = get_model(
        backbone=args.model,
        input_size=args.input_size, 
        reduction=args.reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt
    ).to(device)

    state_dict = torch.load(args.weight_path, map_location=device)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {flops}")
    
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 2)} MB")

    dataset = CustomDataset(root="assets", return_filename=True)

    image_ids = []
    preds = []
    coordinates = []

    for idx in tqdm(range(len(dataset)), desc="Testing on NWPU"):
        start_time = time.time()

        image, image_name = dataset[idx]

        orig_image_path = image_name  # 원본 이미지 경로
        image = image.unsqueeze(0)  # add batch dimension
        orig_shape = image.shape[2:]  # (C, H, W)에서 H, W 추출
        image = image.to(device)  # add batch dimension

        print(f"dataset image loading : {time.time() - start_time}")
        time_step_2 = time.time()

        with torch.set_grad_enabled(False):


            pred_density = model(image)
            print(f"model에 들어감: {time.time() - time_step_2}")
            
            pred_count = pred_density.sum(dim=(1, 2, 3)).item()
            pred_coords = extract_coordinates(pred_density, orig_shape, threshold=0.45)  # Adjust threshold as needed

        print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 2)} MB")
        end_time = time.time()
        print(f"Inference time: {end_time - start_time} seconds")

        image_ids.append(os.path.basename(image_name).split(".")[0])
        preds.append(pred_count)
        coordinates.append(pred_coords)


        # prediction image SAVE
        save_prediction_image_with_dot(orig_image_path, pred_count, pred_coords, f"result/dot/{image_ids[-1]}_dot_prediction.png")
        save_prediction_image_with_heatmap(orig_image_path, pred_density, f"result/image_with_heatmap/{image_ids[-1]}_heatmap_prediction.png")

    result_dir = os.path.join(current_dir, "nwpu_test_results")
    os.makedirs(result_dir, exist_ok=True)
    weights_dir, weights_name = os.path.split(args.weight_path)
    model_name = os.path.split(weights_dir)[-1]
    result_path = os.path.join(result_dir, f"{model_name}_{weights_name.split('.')[0]}.txt")

    with open(result_path, "w") as f:
        for idx, (image_id, pred, coords) in enumerate(zip(image_ids, preds, coordinates)):
            coords_str = " ".join([f"({x},{y})" for y, x in coords])  # x, y 순서로 좌표 저장
            if idx != len(image_ids) - 1:
                f.write(f"{image_id} {pred} {coords_str}\n")
            else:
                f.write(f"{image_id} {pred} {coords_str}")  # no newline at the end of the file

if __name__ == "__main__":

    args = parser.parse_args()
    args.model = args.model.lower()

    main(args)

# Example usage:
# python test_nwpu.py --model vgg19_ae --truncation 4 --weight_path ./checkpoints/sha/vgg19_ae_448_4_1.0_dmcount_aug/best_mae.pth --device cuda:0