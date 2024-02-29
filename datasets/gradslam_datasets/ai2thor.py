import glob  # 導入 glob 模組，用於查找符合特定規則的文件路徑名
import os  # 導入 os 模組，用於操作系統相關的功能，例如讀取環境變量、操作文件路徑等
from pathlib import Path  # 從 pathlib 模組導入 Path 類，用於操作系統路徑，提供了一種面向對象的方式來處理文件路徑
from typing import Dict, List, Optional, Union  # 從 typing 模組導入類型標註工具，用於標註函數的參數和返回值的類型

import cv2  # 導入 cv2 模組，用於圖像處理，例如讀取圖像、調整圖像大小、保存圖像等
import imageio.v2 as imageio  # 導入 imageio.v2 模組並別名為 imageio，用於讀寫圖像，支持多種圖像格式
import numpy as np  # 導入 numpy 模組並別名為 np，用於數值計算，例如矩陣運算、數學函數等
import torch  # 導入 torch 模組，用於深度學習，提供了張量運算、自動求導等功能
import torch.nn.functional as F  # 導入 torch.nn.functional 模組並別名為 F，用於神經網路的功能，例如激活函數、損失函數等
from natsort import natsorted  # 從 natsort 模組導入 natsorted 函數，用於自然排序，可以正確地對包含數字的字符串進行排序

from .basedataset import GradSLAMDataset  # 從 basedataset 模組導入 GradSLAMDataset 類，這是一個基礎的數據集類，提供了一些基礎的方法


class Ai2thorDataset(GradSLAMDataset):  # 定義 Ai2thorDataset 類，繼承自 GradSLAMDataset 類，這是一個專門用於處理 Ai2thor 數據集的類
    def __init__(  # 定義初始化函數，用於初始化 Ai2thorDataset 類的實例
        self,
        config_dict,  # 配置字典，用於存儲一些配置參數
        basedir,  # 基礎目錄，用於指定數據集的路徑
        sequence,  # 序列，用於指定需要處理的數據集的序列
        stride: Optional[int] = None,  # 步長，用於指定在讀取數據時的步長，預設為 None
        start: Optional[int] = 0,  # 開始位置，用於指定在讀取數據時的開始位置，預設為 0
        end: Optional[int] = -1,  # 結束位置，用於指定在讀取數據時的結束位置，預設為 -1
        desired_height: Optional[int] = 968,  # 期望的高度，用於指定在讀取圖像時需要調整到的高度，預設為 968
        desired_width: Optional[int] = 1296,  # 期望的寬度，用於指定在讀取圖像時需要調整到的寬度，預設為 1296
        load_embeddings: Optional[bool] = False,  # 是否加載嵌入，用於指定是否需要加載嵌入，預設為 False
        embedding_dir: Optional[str] = "embeddings",  # 嵌入目錄，用於指定嵌入的路徑，預設為 "embeddings"
        embedding_dim: Optional[int] = 512,  # 嵌入維度，用於指定嵌入的維度，預設為 512
        **kwargs,  # 其他參數，用於接收其他的關鍵字參數
    ):
        self.input_folder = os.path.join(basedir, sequence)  # 設定輸入文件夾路徑，將基礎目錄和序列組合成完整的路徑
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))  # 獲取顏色圖像的路徑，使用 glob.glob 函數查找所有符合特定規則的文件，然後使用 natsorted 函數進行自然排序
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))  # 獲取深度圖像的路徑，使用 glob.glob 函數查找所有符合特定規則的文件，然後使用 natsorted 函數進行自然排序
        embedding_paths = None  # 初始化嵌入路徑為 None
        if self.load_embeddings:  # 如果需要加載嵌入
            if self.embedding_dir == "embed_semseg":  # 如果嵌入目錄為 "embed_semseg"
                # embed_semseg 是以 uint16 pngs 的形式存儲的
                embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.png"))  # 獲取嵌入的路徑，使用 glob.glob 函數查找所有符合特定規則的文件，然後使用 natsorted 函數進行自然排序
            else:
                embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))  # 獲取嵌入的路徑，使用 glob.glob 函數查找所有符合特定規則的文件，然後使用 natsorted 函數進行自然排序
        return color_paths, depth_paths, embedding_paths  # 返回顏色圖像、深度圖像和嵌入的路徑

    def load_poses(self):  # 定義加載姿態的函數，用於讀取姿態文件並將其轉換為張量
        poses = []  # 初始化姿態列表
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))  # 獲取姿態文件的路徑，使用 glob.glob 函數查找所有符合特定規則的文件，然後使用 natsorted 函數進行自然排序
        for posefile in posefiles:  # 對於每個姿態文件
            _pose = torch.from_numpy(np.loadtxt(posefile))  # 使用 numpy 的 loadtxt 函數讀取姿態文件，然後使用 torch 的 from_numpy 函數將其轉換為張量
            poses.append(_pose)  # 將姿態張量添加到姿態列表中
        return poses  # 返回姿態列表

    def read_embedding_from_file(self, embedding_file_path):  # 定義從文件讀取嵌入的函數
        if self.embedding_dir == "embed_semseg":  # 如果嵌入目錄為 "embed_semseg"
            embedding = imageio.imread(embedding_file_path)  # 使用 imageio 的 imread 函數讀取嵌入圖像
            embedding = cv2.resize(  # 使用 cv2 的 resize 函數調整嵌入圖像的大小
                embedding, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST
            )
            embedding = torch.from_numpy(embedding).long()  # 將嵌入圖像轉換為張量
            embedding = F.one_hot(embedding, num_classes=self.embedding_dim)  # 對嵌入進行 one-hot 編碼
            embedding = embedding.half()  # 將嵌入轉換為半精度浮點數
            embedding = embedding.permute(2, 0, 1)  # 調整嵌入的維度順序，將 (高度, 寬度, 通道數) 調整為 (通道數, 高度, 寬度)
            embedding = embedding.unsqueeze(0)  # 在嵌入的第一個維度上增加一個維度，將 (通道數, 高度, 寬度) 調整為 (批次大小, 通道數, 高度, 寬度)
        else:
            embedding = torch.load(embedding_file_path, map_location="cpu")  # 使用 torch 的 load 函數加載嵌入
        return embedding.permute(0, 2, 3, 1)  # 調整嵌入的維度順序，將 (批次大小, 通道數, 高度, 寬度) 調整為 (批次大小, 高度, 寬度, 通道數)，並返回

