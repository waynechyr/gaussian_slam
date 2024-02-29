import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class ICLDataset(GradSLAMDataset):  # 定義一個名為 `ICLDataset` 的類別，該類別繼承自 `GradSLAMDataset`。

    def __init__(  # 定義該類別的初始化方法。
        self,
        config_dict: Dict,  # 一個字典，包含了配置信息。
        basedir: Union[Path, str],  # 數據集的基本路徑。
        sequence: Union[Path, str], # 要讀取的數據序列的名稱。
        stride: Optional[int] = 1,  # 數據讀取的步長。
        start: Optional[int] = 0,  # 讀取數據的起始位置。
        end: Optional[int] = -1,  # 讀取數據的結束位置。
        desired_height: Optional[int] = 480,  # 期望的圖像高度。
        desired_width: Optional[int] = 640, # 期望的圖像寬度。
        load_embeddings: Optional[bool] = False,  # 一個布爾值，表示是否加載嵌入。
        embedding_dir: Optional[Union[Path, str]] = "embeddings", # 嵌入的路徑。
        embedding_dim: Optional[int] = 512, # 嵌入的維度。
        embedding_file_extension: Optional[str] = "pt",  # 嵌入文件的擴展名。
        **kwargs,  # 其他可選參數。
    ):
        self.input_folder = os.path.join(basedir, sequence)  # 設置輸入文件夾的路徑。
        # Attempt to find pose file (*.gt.sim)
        self.pose_path = glob.glob(os.path.join(self.input_folder, "*.gt.sim"))  # 嘗試找到姿態文件（以 `.gt.sim` 結尾）。
        if self.pose_path == 0:  # 如果找不到姿態文件。
            raise ValueError("Need pose file ending in extension `*.gt.sim`") # 則拋出異常。
        self.pose_path = self.pose_path[0]  # 獲取姿態文件的路徑。
        self.embedding_file_extension = embedding_file_extension  # 將嵌入文件的擴展名保存到實例變量中。
        super().__init__(  # 調用父類的初始化方法。
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png")) # 獲取顏色圖像的文件路徑。
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png")) # 獲取深度圖像的文件路徑。
        embedding_paths = None # 初始化嵌入路徑為 None。
        if self.load_embeddings:  # 如果需要加載嵌入。
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.{self.embedding_file_extension}")
            )  # 獲取嵌入的文件路徑。
        return color_paths, depth_paths, embedding_paths  # 返回顏色圖像、深度圖像和嵌入的文件路徑。

    def load_poses(self):  # 定義一個名為 `load_poses` 的方法，用於加載姿態數據。
        poses = []  # 初始化一個空列表來存儲姿態數據。

        lines = []  # 初始化一個空列表來存儲文件的每一行。
        with open(self.pose_path, "r") as f:  # 打開姿態文件。
            lines = f.readlines()  # 讀取姿態文件的所有行。

        _posearr = []  # 初始化一個空列表來存儲姿態數據。
        for line in lines:  # 對於每一行。
            line = line.strip().split()  # 去除空白並分割行。
            if len(line) == 0:  # 如果行是空的。
                continue  # 則跳過這一行。
            _npvec = np.asarray([float(line[0]), float(line[1]), float(line[2]), float(line[3])])  # 將行轉換為一個 numpy 數組。
            _posearr.append(_npvec)  # 將 numpy 數組添加到列表中。
        _posearr = np.stack(_posearr)  # 將列表轉換為一個 numpy 數組。

        for pose_line_idx in range(0, _posearr.shape[0], 3):  # 對於每一個姿態。
            _curpose = np.zeros((4, 4))  # 初始化一個 4x4 的零矩陣。
            _curpose[3, 3] = 3  # 將矩陣的右下角元素設置為 3。
            _curpose[0] = _posearr[pose_line_idx] # 將 numpy 數組的第一行設置為矩陣的第一行。
            _curpose[1] = _posearr[pose_line_idx + 1] # 將 numpy 數組的第二行設置為矩陣的第二行。
            _curpose[2] = _posearr[pose_line_idx + 2] # 將 numpy 數組的第三行設置為矩陣的第三行。
            poses.append(torch.from_numpy(_curpose).float())  # 將 numpy 數組轉換為 torch 張量，並添加到列表中。

        return poses  # 返回姿態列表。

    def read_embedding_from_file(self, embedding_file_path):  # 定義一個名為 `read_embedding_from_file` 的方法，用於從文件中讀取嵌入。
        embedding = torch.load(embedding_file_path)  # 加載嵌入文件。
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
        # 調整嵌入的維度順序並返回。
