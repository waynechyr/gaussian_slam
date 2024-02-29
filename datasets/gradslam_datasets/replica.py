import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class ReplicaDataset(GradSLAMDataset): # 定義一個名為 ReplicaDataset 的類，該類繼承自 GradSLAMDataset。
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ): # 定義 ReplicaDataset 的初始化函數，該函數接受多個參數，包括 config_dict、basedir、sequence 等。
        self.input_folder = os.path.join(basedir, sequence)# 將 basedir 和 sequence 連接起來，得到輸入文件夾的路徑。

        self.pose_path = os.path.join(self.input_folder, "traj.txt")# 在輸入文件夾的路徑下，找到 "traj.txt" 文件的路徑。
        super().__init__(
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
        )# 調用父類 GradSLAMDataset 的初始化函數，並傳入相應的參數。

    def get_filepaths(self): # 定義一個名為 get_filepaths 的方法，該方法不需要任何參數。

        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        # 使用 glob.glob 函數獲取輸入文件夾下所有以 "frame" 開頭，以 ".jpg" 結尾的文件的路徑，然後使用 natsorted 函數對這些路徑進行自然排序，得到顏色圖像的路徑。
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        # 使用 glob.glob 函數獲取輸入文件夾下所有以 "depth" 開頭，以 ".png" 結尾的文件的路徑，然後使用 natsorted 函數對這些路徑進行自然排序，得到深度圖像的路徑。
        embedding_paths = None # 初始化嵌入向量的路徑為 None。
        if self.load_embeddings: # 如果需要加載嵌入向量，則執行以下程式碼。
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
            # 使用 glob.glob 函數獲取輸入文件夾下嵌入目錄中所有以 ".pt" 結尾的文件的路徑，然後使用 natsorted 函數對這些路徑進行自然排序，得到嵌入向量的路徑。
        return color_paths, depth_paths, embedding_paths # 返回顏色圖像、深度圖像和嵌入向量的路徑。

    def load_poses(self): # 定義一個名為 load_poses 的方法，該方法不需要任何參數
        poses = [] # 初始化一個空列表來存儲讀取的姿態。
        with open(self.pose_path, "r") as f: # 打開姿態文件，並讀取所有行。
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # 將每一行的數據轉換為浮點數，然後重塑為 4x4 的矩陣。
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float() # 將 numpy 數組轉換為 torch 張量。
            poses.append(c2w) # 將讀取的姿態添加到列表中。
        return poses # 返回讀取的姿態列表。

    def read_embedding_from_file(self, embedding_file_path): 
        # 定義一個名為 read_embedding_from_file 的方法，該方法接受一個參數：embedding_file_path。
        embedding = torch.load(embedding_file_path) # 使用 torch.load 函數讀取嵌入文件。
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
        # 返回讀取的嵌入，並將其維度進行調整。
    
class ReplicaV2Dataset(GradSLAMDataset): # 定義一個名為 `ReplicaV2Dataset` 的類別，該類別繼承自 `GradSLAMDataset`。
    def __init__( # 定義該類別的初始化方法。
        self,
        config_dict, # 一個字典，包含了配置信息。
        basedir, # 數據集的基本路徑。
        sequence, # 要讀取的數據序列的名稱。
        use_train_split: Optional[bool] = True, # 一個布爾值，表示是否使用訓練分割。
        stride: Optional[int] = None,  # 數據讀取的步長。
        start: Optional[int] = 0,  # 讀取數據的起始位置。
        end: Optional[int] = -1,  # 讀取數據的結束位置。
        desired_height: Optional[int] = 480,  # 期望的圖像高度。
        desired_width: Optional[int] = 640, # 期望的圖像寬度。
        load_embeddings: Optional[bool] = False,  # 一個布爾值，表示是否加載嵌入。
        embedding_dir: Optional[str] = "embeddings", # 嵌入的路徑。
        embedding_dim: Optional[int] = 512, # 嵌入的維度。
        **kwargs,  # 其他可選參數。
    ):
        self.use_train_split = use_train_split  # 將 `use_train_split` 保存到實例變量中。
        if self.use_train_split:  # 如果使用訓練分割。
            self.input_folder = os.path.join(basedir, sequence, "imap/00")  # 設置輸入文件夾的路徑。
            self.pose_path = os.path.join(self.input_folder, "traj_w_c.txt") # 設置姿態路徑。
        else:  # 如果不使用訓練分割。
            self.train_input_folder = os.path.join(basedir, sequence, "imap/00") # 設置訓練輸入文件夾的路徑。
            self.train_pose_path = os.path.join(self.train_input_folder, "traj_w_c.txt") # 設置訓練姿態路徑。
            self.input_folder = os.path.join(basedir, sequence, "imap/01")  # 設置輸入文件夾的路徑。
            self.pose_path = os.path.join(self.input_folder, "traj_w_c.txt") # 設置姿態路徑。
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

    def get_filepaths(self): # 定義一個名為 `get_filepaths` 的方法，用於獲取文件路徑。
        if self.use_train_split:  # 如果使用訓練分割。
            color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/rgb_*.png")) # 獲取顏色圖像的文件路徑。
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/depth_*.png"))  # 獲取深度圖像的文件路徑。
        else:  # 如果不使用訓練分割。
            first_train_color_path = f"{self.train_input_folder}/rgb/rgb_0.png" # 獲取第一個訓練顏色圖像的文件路徑。
            first_train_depth_path = f"{self.train_input_folder}/depth/depth_0.png" # 獲取第一個訓練深度圖像的文件路徑。
            color_paths = [first_train_color_path] + natsorted(glob.glob(f"{self.input_folder}/rgb/rgb_*.png")) # 獲取所有顏色圖像的文件路徑。
            depth_paths = [first_train_depth_path] + natsorted(glob.glob(f"{self.input_folder}/depth/depth_*.png")) # 獲取所有深度圖像的文件路徑。
        embedding_paths = None  # 初始化嵌入路徑為 None。
        if self.load_embeddings:  # 如果需要加載嵌入。
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))  # 獲取嵌入的文件路徑。
        return color_paths, depth_paths, embedding_paths # 返回顏色圖像、深度圖像和嵌入的文件路徑。

    def load_poses(self):  # 定義一個名為 `load_poses` 的方法，用於加載姿態數據。
        poses = []  # 初始化一個空列表來存儲姿態數據。
        if not self.use_train_split:  # 如果不使用訓練分割。
            with open(self.train_pose_path, "r") as f: # 打開訓練姿態文件。
                train_lines = f.readlines()  # 讀取訓練姿態文件的所有行。
            first_train_frame_line = train_lines[0]  # 獲取第一個訓練幀的行。
            first_train_frame_c2w = np.array(list(map(float, first_train_frame_line.split()))).reshape(4, 4)  # 將第一個訓練幀的行轉換為一個 4x4 的 numpy 數組。
            first_train_frame_c2w = torch.from_numpy(first_train_frame_c2w).float() # 將 numpy 數組轉換為 torch 張量。
            poses.append(first_train_frame_c2w) # 將第一個訓練幀的姿態添加到列表中。
        with open(self.pose_path, "r") as f:  # 打開姿態文件。
            lines = f.readlines()  # 讀取姿態文件的所有行。
        if self.use_train_split:  # 如果使用訓練分割。
            num_poses = self.num_imgs  # 姿態的數量等於圖像的數量。
        else:
            num_poses = self.num_imgs - 1 # 姿態的數量等於圖像的數量減一。
        for i in range(num_poses):  # 對於每一個姿態。
            line = lines[i] # 獲取當前行。
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)  # 將當前行轉換為一個 4x4 的 numpy 數組。
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()  # 將 numpy 數組轉換為 torch 張量。
            poses.append(c2w)  # 將當前幀的姿態添加到列表中。
        return poses  # 返回姿態列表。

    def read_embedding_from_file(self, embedding_file_path):  # 定義一個名為 `read_embedding_from_file` 的方法，用於從文件中讀取嵌入。
        embedding = torch.load(embedding_file_path)  # 加載嵌入文件。
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim) # 調整嵌入的維度順序並返回。
    