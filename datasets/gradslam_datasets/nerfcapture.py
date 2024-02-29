import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


def create_filepath_index_mapping(frames):
    return {frame["file_path"]: index for index, frame in enumerate(frames)}


class NeRFCaptureDataset(GradSLAMDataset):  # 定義一個名為 `NeRFCaptureDataset` 的類別，該類別繼承自 `GradSLAMDataset`。
    def __init__(  # 定義該類別的初始化方法。
        self,
        basedir,  # 數據集的基本路徑。
        sequence, # 要讀取的數據序列的名稱。
        stride: Optional[int] = None,  # 數據讀取的步長。
        start: Optional[int] = 0,  # 讀取數據的起始位置。
        end: Optional[int] = -1,  # 讀取數據的結束位置。
        desired_height: Optional[int] = 1440,  # 期望的圖像高度。
        desired_width: Optional[int] = 1920, # 期望的圖像寬度。
        load_embeddings: Optional[bool] = False, # 一個布爾值，表示是否加載嵌入。
        embedding_dir: Optional[str] = "embeddings", # 嵌入的路徑。
        embedding_dim: Optional[int] = 512,  # 嵌入的維度。
        **kwargs,  # 其他可選參數。
    ):
        self.input_folder = os.path.join(basedir, sequence)  # 設置輸入文件夾的路徑。
        config_dict = {}  # 初始化一個空字典來存儲配置信息。
        config_dict["dataset_name"] = "nerfcapture"  # 將數據集名稱設置為 "nerfcapture"。
        self.pose_path = None  # 初始化姿態路徑為 None。
        
        # Load NeRFStudio format camera & poses data
        self.cams_metadata = self.load_cams_metadata() # 加載相機和姿態數據。
        self.frames_metadata = self.cams_metadata["frames"]  # 獲取幀的元數據。
        self.filepath_index_mapping = create_filepath_index_mapping(self.frames_metadata)  # 創建文件路徑和索引的映射。


        # Load RGB & Depth filepaths
        self.image_names = natsorted(os.listdir(f"{self.input_folder}/rgb")) # 加載 RGB 圖像的文件名。
        self.image_names = [f'rgb/{image_name}' for image_name in self.image_names]  # 獲取 RGB 圖像的完整路徑。

        # Init Intrinsics
        config_dict["camera_params"] = {}  # 初始化相機參數字典。
        config_dict["camera_params"]["png_depth_scale"] = 6553.5 # Depth is in mm
        config_dict["camera_params"]["image_height"] = self.cams_metadata["h"] # 設置圖像的高度。
        config_dict["camera_params"]["image_width"] = self.cams_metadata["w"] # 設置圖像的寬度。
        config_dict["camera_params"]["fx"] = self.cams_metadata["fl_x"]  # 設置焦距 x。
        config_dict["camera_params"]["fy"] = self.cams_metadata["fl_y"]  # 設置焦距 y。
        config_dict["camera_params"]["cx"] = self.cams_metadata["cx"]  # 設置中心點 x。
        config_dict["camera_params"]["cy"] = self.cams_metadata["cy"]  # 設置中心點 y。

        super().__init__( # 調用父類的初始化方法。
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        ) 

    def load_cams_metadata(self): # 定義一個名為 `load_cams_metadata` 的方法，用於加載相機元數據。
        cams_metadata_path = f"{self.input_folder}/transforms.json"  # 設置相機元數據的路徑。
        cams_metadata = json.load(open(cams_metadata_path, "r"))  # 加載相機元數據。
        return cams_metadata # 返回相機元數據。
    
    def get_filepaths(self):  # 定義一個名為 `get_filepaths` 的方法，用於獲取文件路徑。
        base_path = f"{self.input_folder}" # 設置基本路徑。
        color_paths = []  # 初始化一個空列表來存儲顏色圖像的路徑。
        depth_paths = []  # 初始化一個空列表來存儲深度圖像的路徑。
        self.tmp_poses = []  # 初始化一個空列表來存儲臨時姿態。
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()  # 創建一個 4x4 的張量 P。
        for image_name in self.image_names:  # 對於每一個圖像名稱。
            # Search for image name in frames_metadata
            frame_metadata = self.frames_metadata[self.filepath_index_mapping.get(image_name)]  # 在幀的元數據中查找圖像名稱。
            # Get path of image and depth
            color_path = f"{base_path}/{image_name}"  # 獲取顏色圖像的路徑。
            depth_path = f"{base_path}/{image_name.replace('rgb', 'depth')}" # 獲取深度圖像的路徑。
            color_paths.append(color_path)  # 將顏色圖像的路徑添加到列表中。
            depth_paths.append(depth_path)  # 將深度圖像的路徑添加到列表中。
            # Get pose of image in GradSLAM format
            c2w = torch.from_numpy(np.array(frame_metadata["transform_matrix"])).float()  # 獲取圖像的姿態並轉換為張量。
            _pose = P @ c2w @ P.T  # 計算姿態。
            self.tmp_poses.append(_pose)  # 將姿態添加到列表中。
        embedding_paths = None  # 初始化嵌入路徑為 None。
        if self.load_embeddings:  # 如果需要加載嵌入。
            embedding_paths = natsorted(glob.glob(f"{base_path}/{self.embedding_dir}/*.pt"))  # 獲取嵌入的文件路徑。
        return color_paths, depth_paths, embedding_paths  # 返回顏色圖像、深度圖像和嵌入的文件路徑。

    def load_poses(self):  # 定義一個名為 `load_poses` 的方法，用於加載姿態。
        return self.tmp_poses  # 返回臨時姿態。

    def read_embedding_from_file(self, embedding_file_path):  # 定義一個名為 `read_embedding_from_file` 的方法，用於從文件中讀取嵌入。
        print(embedding_file_path)  # 打印嵌入文件的路徑。
        embedding = torch.load(embedding_file_path, map_location="cpu")  # 使用 torch.load 方法從文件中加載嵌入，並將數據映射到 CPU。
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)  # 調整嵌入的維度順序並返回。
