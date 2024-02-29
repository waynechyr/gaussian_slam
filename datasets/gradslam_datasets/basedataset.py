"""
PyTorch dataset classes for GradSLAM v1.0.

The base dataset class now loads one sequence at a time
(opposed to v0.1.0 which loads multiple sequences).

A few parts of this code are adapted from NICE-SLAM
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py
"""

import abc # 導入 abc 模組，用於定義抽象基礎類別 (Abstract Base Classes)
import glob # 導入 glob 模組，用於查找符合特定規則的文件路徑名
import os  # 導入 os 模組，用於操作系統相關的功能，例如讀取環境變量、操作文件路徑等
from pathlib import Path  # 從 pathlib 模組導入 Path 類，用於操作系統路徑，提供了一種面向對象的方式來處理文件路徑
from typing import Dict, List, Optional, Union  # 從 typing 模組導入類型標註工具，用於標註函數的參數和返回值的類型

import cv2  # 導入 cv2 模組，用於圖像處理，例如讀取圖像、調整圖像大小、保存圖像等
import imageio  # 導入 imageio 模組，用於讀寫圖像，支持多種圖像格式
import numpy as np  # 導入 numpy 模組並別名為 np，用於數值計算，例如矩陣運算、數學函數等
import torch  # 導入 torch 模組，用於深度學習，提供了張量運算、自動求導等功能
import yaml  # 導入 yaml 模組，用於讀寫 YAML 文件，YAML 是一種用於數據序列化的語言
from natsort import natsorted  # 從 natsort 模組導入 natsorted 函數，用於自然排序，可以正確地對包含數字的字符串進行排序

from .geometryutils import relative_transformation  # 從 geometryutils 模組導入 relative_transformation 函數，用於計算相對變換
from . import datautils  # 導入 datautils 模組，該模組可能包含一些用於數據處理的工具


def to_scalar(inp: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:  # 定義 to_scalar 函數，用於將輸入轉換為標量
    """
    Convert the input to a scalar
    """
    if isinstance(inp, float): # 如果輸入是浮點數
        return inp  # 直接返回輸入

    if isinstance(inp, np.ndarray):  # 如果輸入是 numpy 陣列
        assert inp.size == 1 # 斷言輸入的大小為 1，如果不為 1，則拋出異常
        return inp.item()  # 返回輸入的元素

    if isinstance(inp, torch.Tensor):  # 如果輸入是張量
        assert inp.numel() == 1  # 斷言輸入的元素數量為 1，如果不為 1，則拋出異常
        return inp.item()  # 返回輸入的元素


def as_intrinsics_matrix(intrinsics):  # 定義 as_intrinsics_matrix 函數，用於獲取內參矩陣的表示
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)  # 創建一個 3x3 的單位矩陣
    K[0, 0] = intrinsics[0]  # 將內參的第一個元素賦值給矩陣的第一個元素
    K[1, 1] = intrinsics[1]  # 將內參的第二個元素賦值給矩陣的第五個元素
    K[0, 2] = intrinsics[2]  # 將內參的第三個元素賦值給矩陣的第三個元素
    K[1, 2] = intrinsics[3]  # 將內參的第四個元素賦值給矩陣的第六個元素
    return K  # 返回矩陣


def from_intrinsics_matrix(K):  # 定義 from_intrinsics_matrix 函數，用於從內參矩陣中獲取 fx、fy、cx、cy
    """
    Get fx, fy, cx, cy from the intrinsics matrix

    return 4 scalars
    """
    fx = to_scalar(K[0, 0])  # 從矩陣中獲取第一個元素，並將其轉換為標量，作為 fx
    fy = to_scalar(K[1, 1])  # 從矩陣中獲取第五個元素，並將其轉換為標量，作為 fy
    cx = to_scalar(K[0, 2])  # 從矩陣中獲取第三個元素，並將其轉換為標量，作為 cx
    cy = to_scalar(K[1, 2])  # 從矩陣中獲取第六個元素，並將其轉換為標量，作為 cy
    return fx, fy, cx, cy  # 返回 fx、fy、cx、cy


def readEXR_onlydepth(filename):  # 定義 readEXR_onlydepth 函數，用於從 EXR 圖像文件中讀取深度數據
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath  # 導入 Imath 模組，該模組提供了一些用於圖像處理的工具
    import OpenEXR as exr  # 導入 OpenEXR 模組並別名為 exr，該模組提供了一些用於讀寫 EXR 圖像的工具

    exrfile = exr.InputFile(filename)  # 使用 exr 的 InputFile 類創建一個 EXR 文件對象
    header = exrfile.header()  # 獲取 EXR 文件的頭部信息
    dw = header["dataWindow"]  # 從頭部信息中獲取數據窗口
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)  # 計算圖像的大小

    channelData = dict()  # 初始化通道數據字典

    for c in header["channels"]:  # 對於頭部信息中的每一個通道
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))  # 讀取該通道的數據，並將其轉換為浮點數
        C = np.fromstring(C, dtype=np.float32)  # 將數據轉換為 numpy 陣列
        C = np.reshape(C, isize)  # 調整 numpy 陣列的形狀

        channelData[c] = C  # 將 numpy 陣列添加到通道數據字典中，鍵為通道名，值為 numpy 陣列

    Y = None if "Y" not in header["channels"] else channelData["Y"]  # 如果頭部信息的通道中不包含 "Y"，則將 Y 設定為 None，否則將 Y 設定為通道數據字典中 "Y" 的值

    return Y  # 返回 Y，即深度數據


class GradSLAMDataset(torch.utils.data.Dataset):  # 定義 GradSLAMDataset 類，該類繼承自 torch.utils.data.Dataset 類，是一個用於處理 GradSLAM 數據集的類
    def __init__(  # 定義初始化函數，用於初始化 GradSLAMDataset 類的實例
        self,
        config_dict,  # 配置字典，用於存儲一些配置參數
        stride: Optional[int] = 1,  # 步長，用於指定在讀取數據時的步長，預設為 1
        start: Optional[int] = 0,  # 開始位置，用於指定在讀取數據時的開始位置，預設為 0
        end: Optional[int] = -1,  # 結束位置，用於指定在讀取數據時的結束位置，預設為 -1
        desired_height: int = 480,  # 期望的高度，用於指定在讀取圖像時需要調整到的高度，預設為 480
        desired_width: int = 640,  # 期望的寬度，用於指定在讀取圖像時需要調整到的寬度，預設為 640
        channels_first: bool = False,  # 是否將通道放在第一個維度，用於指定在讀取圖像時是否需要將通道放在第一個維度，預設為 False
        normalize_color: bool = False,  # 是否正規化顏色，用於指定在讀取圖像時是否需要對顏色進行正規化，預設為 False
        device="cuda:0",  # 設備，用於指定在進行張量運算時需要使用的設備，預設為 "cuda:0"
        dtype=torch.float,  # 數據類型，用於指定在進行張量運算時需要使用的數據類型，預設為 torch.float
        load_embeddings: bool = False,  # 是否加載嵌入，用於指定是否需要加載嵌入，預設為 False
        embedding_dir: str = "feat_lseg_240_320",  # 嵌入目錄，用於指定嵌入的路徑，預設為 "feat_lseg_240_320"
        embedding_dim: int = 512,  # 嵌入維度，用於指定嵌入的維度，預設為 512
        relative_pose: bool = True,  # If True, the pose is relative to the first frame # 是否相對於第一幀的姿態，如果為 True，則姿態是相對於第一幀的，預設為 True
        **kwargs,  # 其他參數，用於接收其他的關鍵字參數
    ):
        super().__init__()  # 調用父類的初始化函數
        self.name = config_dict["dataset_name"]  # 從配置字典中獲取數據集名稱，並設定為實例的屬性
        self.device = device  # 將設備設定為實例的屬性
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]  # 從配置字典中獲取 PNG 深度縮放，並設定為實例的屬性

        self.orig_height = config_dict["camera_params"]["image_height"]  # 從配置字典中獲取圖像高度，並設定為實例的屬性
        self.orig_width = config_dict["camera_params"]["image_width"]  # 從配置字典中獲取圖像寬度，並設定為實例的屬性
        self.fx = config_dict["camera_params"]["fx"]  # 從配置字典中獲取焦距 fx，並設定為實例的屬性
        self.fy = config_dict["camera_params"]["fy"]  # 從配置字典中獲取焦距 fy，並設定為實例的屬性
        self.cx = config_dict["camera_params"]["cx"]  # 從配置字典中獲取主點座標 cx，並設定為實例的屬性
        self.cy = config_dict["camera_params"]["cy"]  # 從配置字典中獲取主點座標 cy，並設定為實例的屬性

        self.dtype = dtype  # 將數據類型設定為實例的屬性

        self.desired_height = desired_height  # 將期望的高度設定為實例的屬性
        self.desired_width = desired_width  # 將期望的寬度設定為實例的屬性
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height  # 計算高度的下採樣比例，並設定為實例的屬性
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width  # 計算寬度的下採樣比例，並設定為實例的屬性
        self.channels_first = channels_first  # 將是否將通道放在第一個維度設定為實例的屬性
        self.normalize_color = normalize_color  # 將是否正規化顏色設定為實例的屬性

        self.load_embeddings = load_embeddings  # 將是否加載嵌入設定為實例的屬性
        self.embedding_dir = embedding_dir  # 將嵌入目錄設定為實例的屬性
        self.embedding_dim = embedding_dim  # 將嵌入維度設定為實例的屬性
        self.relative_pose = relative_pose  # 將是否相對於第一幀的姿態設定為實例的屬性

        self.start = start  # 將開始位置設定為實例的屬性
        self.end = end  # 將結束位置設定為實例的屬性
        if start < 0:  # 如果開始位置小於 0
            raise ValueError("start must be positive. Got {0}.".format(stride))  # 拋出異常
        if not (end == -1 or end > start):  # 如果結束位置不等於 -1 且結束位置小於或等於開始位置
            raise ValueError("end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start))  # 拋出異常

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )  # 從配置字典中獲取失真參數，如果配置字典中包含 "distortion"，則將其轉換為 numpy 陣列，否則設定為 None
        self.crop_size = (
            config_dict["camera_params"]["crop_size"] if "crop_size" in config_dict["camera_params"] else None
        )  # 從配置字典中獲取裁剪大小，如果配置字典中包含 "crop_size"，則直接獲取其值，否則設定為 None

        self.crop_edge = None  # 初始化裁剪邊緣為 None
        if "crop_edge" in config_dict["camera_params"].keys():  # 如果配置字典的 "camera_params" 中包含 "crop_edge"
            self.crop_edge = config_dict["camera_params"]["crop_edge"]  # 從配置字典中獲取裁剪邊緣，並設定為實例的屬性

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()  # 獲取顏色圖像、深度圖像和嵌入的路徑
        if len(self.color_paths) != len(self.depth_paths):  # 如果顏色圖像的數量不等於深度圖像的數量
            raise ValueError("Number of color and depth images must be the same.")  # 拋出異常
        if self.load_embeddings:  # 如果需要加載嵌入
            if len(self.color_paths) != len(self.embedding_paths):  # 如果顏色圖像的數量不等於嵌入的數量
                raise ValueError("Mismatch between number of color images and number of embedding files.")  # 拋出異常
        self.num_imgs = len(self.color_paths)  # 計算顏色圖像的數量，並設定為實例的屬性
        self.poses = self.load_poses()  # 加載姿態，並設定為實例的屬性

        if self.end == -1:  # 如果結束位置為 -1
            self.end = self.num_imgs  # 將結束位置設定為圖像的數量

        self.color_paths = self.color_paths[self.start : self.end : stride]  # 從開始位置到結束位置，每隔 stride 個位置，選取一個顏色圖像的路徑
        self.depth_paths = self.depth_paths[self.start : self.end : stride]  # 從開始位置到結束位置，每隔 stride 個位置，選取一個深度圖像的路徑
        if self.load_embeddings:  # 如果需要加載嵌入
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]  # 從開始位置到結束位置，每隔 stride 個位置，選取一個嵌入的路徑
        self.poses = self.poses[self.start : self.end : stride]  # 從開始位置到結束位置，每隔 stride 個位置，選取一個姿態
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]  # 創建一個從 0 到圖像數量的張量，然後從開始位置到結束位置，每隔 stride 個位置，選取一個索引
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)  # 更新圖像的數量，設定為顏色圖像的數量

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        self.poses = torch.stack(self.poses) # 將姿態列表轉換為張量，並設定為實例的屬性
        if self.relative_pose:  # 如果姿態是相對於第一幀的
            self.transformed_poses = self._preprocess_poses(self.poses)  # 對姿態進行預處理，並設定為實例的屬性
        else:
            self.transformed_poses = self.poses  # 將姿態設定為實例的屬性

    def __len__(self):  # 定義 __len__ 方法，用於獲取數據集的長度
        return self.num_imgs  # 返回圖像的數量

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError  # 提示需要在子類中實現該方法

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError  # 提示需要在子類中實現該方法

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        ) # 將顏色圖像調整到期望的大小
        if self.normalize_color:
            color = datautils.normalize_image(color)  # 如果需要正規化顏色，則對顏色圖像進行正規化
        if self.channels_first:
            color = datautils.channels_first(color)  # 如果需要將通道放在第一個維度，則將顏色圖像的通道放在第一個維度
        return color  # 返回處理後的顏色圖像

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )  # 將深度圖像調整到期望的大小
        depth = np.expand_dims(depth, -1)  # 為深度圖像添加一個通道維度
        if self.channels_first:
            depth = datautils.channels_first(depth)  # 如果需要將通道放在第一個維度，則將深度圖像的通道放在第一個維度
        return depth / self.png_depth_scale  # 返回處理後的深度圖像，並將深度值轉換為米

    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )  # 將第一個姿態設定為單位矩陣，並計算所有其他姿態的相對齊次變換

    def get_cam_K(self):
        """
        Return camera intrinsics matrix K

        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        """
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])  # 獲取內參矩陣的表示
        K = torch.from_numpy(K)  # 將內參矩陣轉換為張量
        return K  # 返回內參矩陣

    def read_embedding_from_file(self, embedding_path: str):
        """
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        """
        raise NotImplementedError  # 提示需要在子類中實現該方法
    """
    這段程式碼定義了一個名為 __getitem__ 的方法，
    該方法用於獲取指定索引的數據。該方法首先讀取顏色圖像和深度圖像，然後對它們進行預處理，
    包括調整大小、去失真等。然後，該方法獲取內參矩陣和姿態，並將它們轉換為張量。
    最後，該方法返回一個包含顏色圖像、深度圖像、內參矩陣和姿態的元組。如果需要加載嵌入，則該方法還會讀取嵌入並將其添加到返回的元組中
    """
    def __getitem__(self, index):  # 定義 __getitem__ 方法，用於獲取指定索引的數據
        color_path = self.color_paths[index]  # 獲取顏色圖像的路徑
        depth_path = self.depth_paths[index]  # 獲取深度圖像的路徑
        color = np.asarray(imageio.imread(color_path), dtype=float)  # 讀取顏色圖像，並將其轉換為 numpy 陣列
        color = self._preprocess_color(color)  # 對顏色圖像進行預處理
        if ".png" in depth_path:  # 如果深度圖像的路徑包含 ".png"
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64) # 讀取深度圖像，並將其轉換為 numpy 陣列
        elif ".exr" in depth_path:  # 如果深度圖像的路徑包含 ".exr"
            depth = readEXR_onlydepth(depth_path)  # 讀取深度數據 

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])  # 獲取內參矩陣的表示
        if self.distortion is not None:  # 如果失真參數不為 None
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)  # 對顏色圖像進行去失真處理

        color = torch.from_numpy(color)  # 將顏色圖像轉換為張量
        K = torch.from_numpy(K)  # 將內參矩陣轉換為張量

        depth = self._preprocess_depth(depth)  # 對深度圖像進行預處理
        depth = torch.from_numpy(depth)  # 將深度圖像轉換為張量

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)  # 對內參矩陣進行縮放
        intrinsics = torch.eye(4).to(K)  # 創建一個 4x4 的單位矩陣，並將其轉換到與內參矩陣相同的設備上
        intrinsics[:3, :3] = K  # 將內參矩陣的值賦值給單位矩陣的前三行和前三列

        pose = self.transformed_poses[index]  # 獲取指定索引的姿態

        if self.load_embeddings:  # 如果需要加載嵌入
            embedding = self.read_embedding_from_file(self.embedding_paths[index])  # 讀取嵌入
            return (
                color.to(self.device).type(self.dtype),  # 將顏色圖像轉換到指定的設備上，並將其數據類型轉換為指定的數據類型
                depth.to(self.device).type(self.dtype),  # 將深度圖像轉換到指定的設備上，並將其數據類型轉換為指定的數據類型
                intrinsics.to(self.device).type(self.dtype),  # 將內參矩陣轉換到指定的設備上，並將其數據類型轉換為指定的數據類型
                pose.to(self.device).type(self.dtype),  # 將姿態轉換到指定的設備上，並將其數據類型轉換為指定的數據類型
                embedding.to(self.device),  # Allow embedding to be another dtype  # 將嵌入轉換到指定的設備上
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),  # 將顏色圖像轉換到指定的設備上，並將其數據類型轉換為指定的數據類型
            depth.to(self.device).type(self.dtype),  # 將深度圖像轉換到指定的設備上，並將其數據類型轉換為指定的數據類型
            intrinsics.to(self.device).type(self.dtype),  # 將內參矩陣轉換到指定的設備上，並將其數據類型轉換為指定的數據類型
            pose.to(self.device).type(self.dtype),  # 將姿態轉換到指定的設備上，並將其數據類型轉換為指定的數據類型
            # self.retained_inds[index].item(),
        )
