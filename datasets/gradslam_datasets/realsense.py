import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class RealsenseDataset(GradSLAMDataset):  # 定義一個名為 `RealsenseDataset` 的類別，該類別繼承自 `GradSLAMDataset`。
    """
    Dataset class to process depth images captured by realsense camera on the tabletop manipulator
    """

    def __init__(  # 定義該類別的初始化方法。
        self,
        config_dict,  # 一個字典，包含了配置信息。
        basedir,  # 數據集的基本路徑。
        sequence,  # 要讀取的數據序列的名稱。
        stride: Optional[int] = None,  # 數據讀取的步長。
        start: Optional[int] = 0,  # 讀取數據的起始位置。
        end: Optional[int] = -1, # 讀取數據的結束位置。
        desired_height: Optional[int] = 480,  # 期望的圖像高度。
        desired_width: Optional[int] = 640,  # 期望的圖像寬度。
        load_embeddings: Optional[bool] = False,  # 一個布爾值，表示是否加載嵌入。
        embedding_dir: Optional[str] = "embeddings",  # 嵌入的路徑。
        embedding_dim: Optional[int] = 512,  # 嵌入的維度。
        **kwargs,  # 其他可選參數。
    ):
        self.input_folder = os.path.join(basedir, sequence)  # 設置輸入文件夾的路徑。
        # only poses/images/depth corresponding to the realsense_camera_order are read/used
        self.pose_path = os.path.join(self.input_folder, "poses") #設置姿態路徑
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

    def get_filepaths(self):  # 定義一個名為 `get_filepaths` 的方法，用於獲取文件路徑。
        color_paths = natsorted(glob.glob(os.path.join(self.input_folder, "rgb", "*.jpg")))  # 獲取顏色圖像的文件路徑。
        depth_paths = natsorted(glob.glob(os.path.join(self.input_folder, "depth", "*.png")))  # 獲取深度圖像的文件路徑。
        embedding_paths = None  # 初始化嵌入路徑為 None。
        if self.load_embeddings:  # 如果需要加載嵌入。
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))  # 獲取嵌入的文件路徑。
        return color_paths, depth_paths, embedding_paths  # 返回顏色圖像、深度圖像和嵌入的文件路徑。

    def load_poses(self):  # 定義一個名為 `load_poses` 的方法，用於加載姿態。
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))  # 獲取姿態文件的路徑。
        poses = []  # 初始化一個空列表來存儲姿態。
        P = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).float()  # 創建一個 4x4 的張量 P。
        for posefile in posefiles:  # 對於每一個姿態文件。
            c2w = torch.from_numpy(np.load(posefile)).float() # 加載姿態文件並轉換為張量。
            _R = c2w[:3, :3]  # 獲取旋轉矩陣。
            _t = c2w[:3, 3]  # 獲取平移向量。
            _pose = P @ c2w @ P.T # 計算姿態。
            poses.append(_pose)  # 將姿態添加到列表中。
        return poses  # 返回姿態列表。

    def read_embedding_from_file(self, embedding_file_path):  # 定義一個名為 `read_embedding_from_file` 的方法，用於從文件中讀取嵌入。
        embedding = torch.load(embedding_file_path)  # 加載嵌入文件。
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)  # 調整嵌入的維度順序並返回。