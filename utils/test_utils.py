import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 임계값을 기반으로 좌표를 추출하는 함수
def extract_coordinates(density_map, orig_shape, threshold=0.5):
    density_map = density_map.squeeze().cpu().numpy()
    y_coords, x_coords = np.where(density_map >= threshold)
    scale_y = orig_shape[0] / density_map.shape[0]
    scale_x = orig_shape[1] / density_map.shape[1]
    y_coords = (y_coords * scale_y).astype(int)
    x_coords = (x_coords * scale_x).astype(int)
    return list(zip(y_coords, x_coords))

def save_prediction_image_with_dot(image_path, prediction, coords, filename):
    # 원본 이미지를 불러오기
    # image_path = "data/nwpu/test/images/" + image_path
    image = Image.open(image_path).convert("RGB")
    
    draw = ImageDraw.Draw(image)

    # 얼굴 점 좌표를 점으로 표시
    for coord in coords:
        draw.ellipse((coord[1]-2, coord[0]-2, coord[1]+2, coord[0]+2), fill='red', outline='red')

    
    # 이미지 저장
    image.save(filename)

def save_prediction_image_with_heatmap(image_path, density_map, save_path):
    # 원본 이미지를 불러오기
    # image_path = "data/nwpu/test/images/" + image_path
    image = Image.open(image_path).convert("RGB")

    # 히트맵 생성
    density_map = density_map.squeeze().cpu().numpy()
    plt.imshow(density_map, cmap='jet', alpha=0.5)
    plt.axis('off')

    # 히트맵 이미지를 PNG 형식으로 저장
    heatmap_path = save_path.replace("image_with_heatmap", "heatmap")
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 히트맵 이미지를 로드하여 새로운 이미지 위에 붙여넣기
    heatmap_image = Image.open(heatmap_path).convert("RGBA")
    heatmap_image = heatmap_image.resize(image.size, Image.Resampling.LANCZOS)

    # 히트맵을 반투명하게 설정
    alpha = 0.9  # 투명도 설정 (0.0: 완전 투명, 1.0: 완전 불투명)
    heatmap_image = Image.blend(image.convert("RGBA"), heatmap_image, alpha)

    # 최종 이미지 저장
    heatmap_image.convert("RGB").save(save_path)

    