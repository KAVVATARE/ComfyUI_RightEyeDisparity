import torch
import numpy as np
import os
import sys
from typing import Union, List
from PIL import Image
import cv2

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import stereoimage_generation as sig
from comfy.utils import ProgressBar

def tensor2np(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:  # Batch of images
        tensor = tensor[0]  # Assuming we take the first image in the batch
    np_array = tensor.cpu().numpy()
    np_array = np.clip(255.0 * np_array, 0, 255).astype(np.uint8)
    if np_array.shape[0] == 3:  # Convert from (3, H, W) to (H, W, 3)
        np_array = np_array.transpose(1, 2, 0)
    return np_array
    
def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)

class VideoRightEyeNode:
    """
    動画用の右目視差生成ノード
    入力された動画フレームを左目として、右目用の視差動画のみを生成
    メモリ効率を重視したシンプルな実装
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 左目動画フレーム（バッチ）
                "depth_maps": ("IMAGE",),  # 各フレームの深度マップ
                "fill_technique": ([
                    'No fill', 'No fill - Reverse projection', 'Imperfect fill - Hybrid Edge', 'Fill - Naive',
                    'Fill - Naive interpolating', 'Fill - Polylines Soft', 'Fill - Polylines Sharp',
                    'Fill - Post-fill', 'Fill - Reverse projection with Post-fill', 'Fill - Hybrid Edge with fill'
                ], {"default": "Fill - Polylines Soft"}),
            },
            "optional": {
                "divergence": ("FLOAT", {"default": 3.5, "min": 0.05, "max": 15, "step": 0.01}), 
                "separation": ("FLOAT", {"default": 0, "min": -5, "max": 5, "step": 0.01}),
                "stereo_balance": ("FLOAT", {"default": 0, "min": -0.95, "max": 0.95, "step": 0.05}),
                "stereo_offset_exponent": ("FLOAT", {"default": 2, "min": 1, "max": 2, "step": 1}),
                "depth_blur_sigma": ("FLOAT", {"default": 0, "min": 0, "max": 10, "step": 0.1}),
                "depth_blur_edge_threshold": ("FLOAT", {"default": 40, "min": 0.1, "max": 100, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)  # 右目動画のみを出力
    FUNCTION = "generate_right_eye"
    CATEGORY = "image/3d"

    def generate_right_eye(self, images, depth_maps, divergence=3.5, separation=0, 
                          stereo_balance=0, stereo_offset_exponent=2, fill_technique="Fill - Polylines Soft", 
                          depth_blur_sigma=0, depth_blur_edge_threshold=40):
        
        fill_technique_mapping = {
            'No fill': 'none',
            'No fill - Reverse projection': 'inverse',
            'Imperfect fill - Hybrid Edge': 'hybrid_edge',
            'Fill - Naive': 'naive',
            'Fill - Naive interpolating': 'naive_interpolating',
            'Fill - Polylines Soft': 'polylines_soft',
            'Fill - Polylines Sharp': 'polylines_sharp',
            'Fill - Post-fill': 'none_post',
            'Fill - Reverse projection with Post-fill': 'inverse_post',
            'Fill - Hybrid Edge with fill': 'hybrid_edge_plus'
        }
        
        fill_technique = fill_technique_mapping.get(fill_technique, 'none')
        
        right_eye_frames = []
        total_frames = len(images)
        pbar = ProgressBar(total_frames)
        
        for i in range(total_frames):
            # 現在のフレームを処理
            left_frame = tensor2np(images[i:i+1])
            depth_map = tensor2np(depth_maps[i:i+1])
            
            # 深度マップをグレースケールに変換
            if len(depth_map.shape) == 3 and depth_map.shape[2] == 3:
                depth_map = np.dot(depth_map[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
            # サイズを合わせる
            if left_frame.shape[:2] != depth_map.shape:
                depth_map = np.array(Image.fromarray(depth_map).resize((left_frame.shape[1], left_frame.shape[0])))
            
            # 右目画像を生成（divergenceを負にして右目用に）
            output = sig.create_stereoimages(
                left_frame, depth_map, 
                -1.0 * divergence,  # 右目用に反転
                separation,  
                ['left-only'],  # 単一画像として出力
                stereo_balance, 
                stereo_offset_exponent, 
                fill_technique, 
                depth_blur_sigma, 
                depth_blur_edge_threshold,
                direction_aware_depth_blur=False,  # メモリ節約のため無効
                return_modified_depth=False  # 深度マップは返さない
            )
            
            # 結果を取得（単一画像のリスト）
            right_frame = np.array(output[0])
            
            # テンソルに変換して追加
            right_eye_frames.append(np2tensor(right_frame))
            
            pbar.update(1)
        
        # すべてのフレームを結合して返す
        return (torch.cat(right_eye_frames),)


# 既存のRightEyeImageNodeもそのまま残す（単一画像用）
class RightEyeImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "fill_technique": ([
                    'No fill', 'No fill - Reverse projection', 'Imperfect fill - Hybrid Edge', 'Fill - Naive',
                    'Fill - Naive interpolating', 'Fill - Polylines Soft', 'Fill - Polylines Sharp',
                    'Fill - Post-fill', 'Fill - Reverse projection with Post-fill', 'Fill - Hybrid Edge with fill'
                ], {"default": "Fill - Polylines Soft"}),
            },
            "optional": {
                "divergence": ("FLOAT", {"default": 3.5, "min": 0.05, "max": 15, "step": 0.01}), 
                "separation": ("FLOAT", {"default": 0, "min": -5, "max": 5, "step": 0.01}),
                "stereo_balance": ("FLOAT", {"default": 0, "min": -0.95, "max": 0.95, "step": 0.05}),
                "stereo_offset_exponent": ("FLOAT", {"default": 2, "min": 1, "max": 2, "step": 1}),
                "depth_blur_sigma": ("FLOAT", {"default": 0, "min": 0, "max": 10, "step": 0.1}),
                "depth_blur_edge_threshold": ("FLOAT", {"default": 40, "min": 0.1, "max": 100, "step": 0.1})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("right_eye_image", "modified_depthmap_right", "right_no_fill_mask")
    FUNCTION = "generate_right_eye"

    def generate_right_eye(self, image, depth_map, divergence=3.5, separation=0, 
                 stereo_balance=0, stereo_offset_exponent=2, fill_technique="Fill - Polylines Soft", 
                 depth_blur_sigma=0, depth_blur_edge_threshold=40):
        
        fill_technique_mapping = {
            'No fill': 'none',
            'No fill - Reverse projection': 'inverse',
            'Imperfect fill - Hybrid Edge': 'hybrid_edge',
            'Fill - Naive': 'naive',
            'Fill - Naive interpolating': 'naive_interpolating',
            'Fill - Polylines Soft': 'polylines_soft',
            'Fill - Polylines Sharp': 'polylines_sharp',
            'Fill - Post-fill': 'none_post',
            'Fill - Reverse projection with Post-fill': 'inverse_post',
            'Fill - Hybrid Edge with fill': 'hybrid_edge_plus'
        }
        
        fill_technique = fill_technique_mapping.get(fill_technique, 'none')
          
        right_images_final = []
        modified_depthmap_final = []
        mask_final = []
        total_steps = len(image)
        pbar = ProgressBar(total_steps)
        
        for i in range(len(image)):
            img = tensor2np(image[i:i+1])
            dm = tensor2np(depth_map[i:i+1])
        
            if len(dm.shape) == 3 and dm.shape[2] == 3:
                dm = np.dot(dm[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
            if img.shape[:2] != dm.shape:
                dm = np.array(Image.fromarray(dm).resize((img.shape[1], img.shape[0])))
            
            output = sig.create_stereoimages(img, dm, -1.0 * divergence, separation,  
                                             ['left-only'], stereo_balance, stereo_offset_exponent, 
                                             fill_technique, depth_blur_sigma, depth_blur_edge_threshold,
                                             direction_aware_depth_blur=True, return_modified_depth=True)
            
            if len(output) == 3:
                results, _, right_modified_depthmap = output
                modified_depthmap = right_modified_depthmap
            else:
                results, modified_depthmap = output
            
            right_img = results[0]
            right_img_tensor = np2tensor(np.array(right_img))
            right_images_final.append(right_img_tensor)
            
            if isinstance(modified_depthmap, Image.Image):
                modified_depthmap_np = np.array(modified_depthmap)
            else:
                modified_depthmap_np = modified_depthmap
            modified_depthmap_tensor = np2tensor(modified_depthmap_np)
            modified_depthmap_final.append(modified_depthmap_tensor)
            
            mask = self.generate_mask(right_img)
            mask_final.append(mask)
            
            pbar.update(1)
        
        return (torch.cat(right_images_final), torch.cat(modified_depthmap_final), torch.cat(mask_final))

    def generate_mask(self, image):
        if isinstance(image, Image.Image):
            np_img = np.array(image)
        else:
            np_img = image
            
        if len(np_img.shape) == 3 and np_img.shape[2] >= 3:
            mask = (np_img.sum(axis=-1) == 0).astype(np.uint8) * 255
        else:
            mask = (np_img == 0).astype(np.uint8) * 255
            
        return np2tensor(mask)


NODE_CLASS_MAPPINGS = {
    "VideoRightEyeNode": VideoRightEyeNode,
    "RightEyeImageNode": RightEyeImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoRightEyeNode": "Video Right Eye Disparity",
    "RightEyeImageNode": "Right Eye Image Node"
}