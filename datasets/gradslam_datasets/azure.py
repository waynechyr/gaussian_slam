import glob  # 導入 glob 模組，用於查找符合特定規則的文件路徑名
import os  # 導入 os 模組，用於操作系統相關的功能，例如讀取環境變量、操作文件路徑等
from pathlib import Path  # 從 pathlib 模組導入 Path 類，用於操作系統路徑，提供了一種面向對象的方式來處理文件路徑
from typing import Dict, List, Optional, Union  # 從 typing 模組導入類型標註工具，用於標註函數的參數和返回值的類型

import numpy as np  # 導入 numpy 模組並別名為 np，用於數值計算，例如矩陣運算、數學函數等
import torch  # 導入 torch 模組，用於深度學習，提供了張量運算、自動求導等功能
from natsort import natsorted  # 從 natsort 模組導入 natsorted 函數，用於自然排序，可以正確地對包含數字的字符串進行排序

from .basedataset import GradSLAMDataset  # 從 basedataset 模組導入 GradSLAMDataset 類，這是一個基礎的數據集類，提供了一些基礎的方法


class AzureKinectDataset(GradSLAMDataset):  # 定義 AzureKinectDataset 類，繼承自 GradSLAMDataset 類，這是一個專門用於處理 Azure Kinect 數據集的類
    def __init__(  # 定義初始化函數，用於初始化 AzureKinectDataset 類的實例
        self,
        config_dict,  # 配置字典，用於存儲一些配置參數
        basedir,  # 基礎目錄，用於指定數據集的路徑
        sequence,  # 序列，用於指定需要處理的數據集的序列
        stride: Optional[int] = None,  # 步長，用於指定在讀取數據時的步長，預設為 None
        start: Optional[int] = 0,  # 開始位置，用於指定在讀取數據時的開始位置，預設為 0
        end: Optional[int] = -1,  # 結束位置，用於指定在讀取數據時的結束位置，預設為 -1
        desired_height: Optional[int] = 480,  # 期望的高度，用於指定在讀取圖像時需要調整到的高度，預設為 480
        desired_width: Optional[int] = 640,  # 期望的寬度，用於指定在讀取圖像時需要調整到的寬度，預設為 640
        load_embeddings: Optional[bool] = False,  # 是否加載嵌入，用於指定是否需要加載嵌入，預設為 False
        embedding_dir: Optional[str] = "embeddings",  # 嵌入目錄，用於指定嵌入的路徑，預設為 "embeddings"
        embedding_dim: Optional[int] = 512,  # 嵌入維度，用於指定嵌入的維度，預設為 512
        **kwargs,  # 其他參數，用於接收其他的關鍵字參數
    ):
        self.input_folder = os.path.join(basedir, sequence)  # 設定輸入文件夾路徑，將基礎目錄和序列組合成完整的路徑
        self.pose_path = None  # 初始化姿態路徑為 None

        # # check if a file named 'poses_global_dvo.txt' exists in the basedir / sequence folder
        # if os.path.isfile(os.path.join(basedir, sequence, "poses_global_dvo.txt")):
        #     self.pose_path = os.path.join(basedir, sequence, "poses_global_dvo.txt")

        if "odomfile" in kwargs.keys():  # 如果關鍵字參數中包含 "odomfile"
            self.pose_path = os.path.join(self.input_folder, kwargs["odomfile"])  # 設定姿態路徑，將輸入文件夾路徑和 "odomfile" 的值組合成完整的路徑
        super().__init__(  # 調用父類的初始化函數，將配置字典和其他參數傳遞給父類的初始化函數
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

    def get_filepaths(self):  # 定義獲取文件路徑的函數，用於獲取顏色圖像、深度圖像和嵌入的路徑
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))  # 獲取顏色圖像的路徑，使用 glob.glob 函數查找所有符合特定規則的文件，然後使用 natsorted 函數進行自然排序
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))  # 獲取深度圖像的路徑，使用 glob.glob 函數查找所有符合特定規則的文件，然後使用 natsorted 函數進行自然排序
        embedding_paths = None  # 初始化嵌入路徑為 None
        if self.load_embeddings:  # 如果需要加載嵌入
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))  # 獲取嵌入的路徑，使用 glob.glob 函數查找所有符合特定規則的文件，然後使用 natsorted 函數進行自然排序
        return color_paths, depth_paths, embedding_paths  # 返回顏色圖像、深度圖像和嵌入的路徑

    def load_poses(self):  # 定義加載姿態的函數，用於讀取姿態文件並將其轉換為張量
        if self.pose_path is None:  # 如果姿態路徑為 None
            print("WARNING: Dataset does not contain poses. Returning identity transform.")  # 打印警告信息
            return [torch.eye(4).float() for _ in range(self.num_imgs)]  # 返回一個列表，列表中的每個元素都是 4x4 的單位矩陣，列表的長度等於圖像的數量
        else:  # 如果姿態路徑不為 None
            # Determine whether the posefile ends in ".log"
            # a .log file has the following format for each frame
            # frame_idx frame_idx+1
            # row 1 of 4x4 transform
            # row 2 of 4x4 transform
            # row 3 of 4x4 transform
            # row 4 of 4x4 transform
            # [repeat for all frames]
            #
            # on the other hand, the "poses_o3d.txt" or "poses_dvo.txt" files have the format
            # 16 entries of 4x4 transform
            # [repeat for all frames]
            if self.pose_path.endswith(".log"):  # 如果姿態文件的路徑以 ".log" 結尾
                # print("Loading poses from .log format")
                poses = []  # 初始化姿態列表
                lines = None  # 初始化行列表為 None
                with open(self.pose_path, "r") as f:  # 打開姿態文件
                    lines = f.readlines()  # 讀取姿態文件的所有行
                if len(lines) % 5 != 0:  # 如果行數不是 5 的倍數
                    raise ValueError(  # 拋出異常
                        "Incorrect file format for .log odom file " "Number of non-empty lines must be a multiple of 5"
                    )
                num_lines = len(lines) // 5  # 計算行數除以 5 的結果
                for i in range(0, num_lines):  # 對於每個索引
                    _curpose = []  # 初始化當前姿態列表
                    _curpose.append(list(map(float, lines[5 * i + 1].split())))  # 讀取第一行的數據，將其轉換為浮點數，並添加到當前姿態列表中
                    _curpose.append(list(map(float, lines[5 * i + 2].split())))  # 讀取第二行的數據，將其轉換為浮點數，並添加到當前姿態列表中
                    _curpose.append(list(map(float, lines[5 * i + 3].split())))  # 讀取第三行的數據，將其轉換為浮點數，並添加到當前姿態列表中
                    _curpose.append(list(map(float, lines[5 * i + 4].split())))  # 讀取第四行的數據，將其轉換為浮點數，並添加到當前姿態列表中
                    _curpose = np.array(_curpose).reshape(4, 4)  # 將當前姿態列表轉換為 numpy 陣列，並調整其形狀為 4x4
                    poses.append(torch.from_numpy(_curpose))  # 將當前姿態轉換為張量，並添加到姿態列表中
            else:  # 如果姿態文件的路徑不以 ".log" 結尾
                poses = []  # 初始化姿態列表
                lines = None  # 初始化行列表為 None
                with open(self.pose_path, "r") as f:  # 打開姿態文件
                    lines = f.readlines()  # 讀取姿態文件的所有行
                for line in lines:  # 對於每一行
                    if len(line.split()) == 0: # 如果該行分割後的長度為 0，即該行為空行
                        continue # 繼續處理下一行
                    c2w = np.array(list(map(float, line.split()))).reshape(4, 4) # 將該行的數據分割並轉換為浮點數，然後轉換為 numpy 陣列，並調整其形狀為 4x4
                    poses.append(torch.from_numpy(c2w))  # 將 numpy 陣列轉換為張量，並添加到姿態列表中
            return poses  # 返回姿態列表

    def read_embedding_from_file(self, embedding_file_path):  # 定義從文件讀取嵌入的函數
        embedding = torch.load(embedding_file_path)  # 使用 torch 的 load 函數加載嵌入
        return embedding  # .permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)  # 返回嵌入
