import cv2
import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from argparse import ArgumentParser
import os, json
import numpy as np

current_dir = os.path.abspath(os.path.dirname(__file__))

from models import CLIP_EBC, get_model
from utils import get_config

parser = ArgumentParser(description="Test a trained model on the Custom test set.")
# Parameters for model
parser.add_argument("--model", type=str, default="clip-resnet50", help="The model to train.")
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
parser.add_argument("--video_input_path", type=str, required=True, help="The path to the video input.")
parser.add_argument("--video_output_path", type=str, required=True, help="The path to the video output.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--device", type=str, default="cuda", help="The device to use for evaluation.")
parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for the data loader.")

data_process = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

    cap = cv2.VideoCapture(args.video_input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps

    target_fps = 10
    frame_interval = int(original_fps / target_fps)

    print(f"Original FPS: {original_fps}")
    print(f"Target FPS: {target_fps}")
    print(f"Frame Interval: {frame_interval}")
    print(f"Total frames: {frame_count}")
    print(f"Duration (s): {duration}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(args.video_output_path, fourcc, original_fps, (width*2, height))

    frames = []
    overlay_frames = []
    frame_id = 0
    frame_id_predict = 0

    print(f"Frame interval : {frame_interval}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        if frame_id % frame_interval == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = data_process(image).to(device)
            inputs = inputs.unsqueeze(0)

            with torch.no_grad():
                outputs = model(inputs)
                density_map_resized = F.interpolate(outputs, size=(height, width), mode='bilinear', align_corners=False)
                density_map_resized = density_map_resized.squeeze().cpu().numpy()

                density_map_resized = (density_map_resized - density_map_resized.min()) / (density_map_resized.max() - density_map_resized.min())
                density_map_resized = (density_map_resized * 255).astype(np.uint8)

                density_map_colormap = cv2.applyColorMap(density_map_resized, cv2.COLORMAP_JET)
                overlay_frame = cv2.addWeighted(frame, 0.25, density_map_colormap, 0.75, 0)
                cv2.putText(overlay_frame, f'People Count: {outputs.sum().item():.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                overlay_frames.append(overlay_frame)
                frame_id_predict += 1
            # print(f"frame for predict : {frame_id_predict}")
            
        frame_id += 1
        print(f"Total frame : {frame_count} | Current frame : {frame_id}")
    cap.release()

    # 예측 프레임의 간격에 맞춰 프레임 재조정
    resampled_overlay_frames = []
    for i in range(frame_count):
        index = int(i * len(overlay_frames) / frame_count)
        resampled_overlay_frames.append(overlay_frames[index])

    for i in range(frame_count):
        overlay_frame = resampled_overlay_frames[i]
        combined_frame = np.hstack((frames[i], overlay_frame))
        out.write(combined_frame)

    out.release()

    print("Video processing completed.")

if __name__ == "__main__":
    args = parser.parse_args()
    args.model = args.model.lower()
    main(args)
