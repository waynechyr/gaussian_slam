import copy  # 導入 copy 模組
import warnings  # 導入 warnings 模組
from collections import OrderedDict  # 從 collections 模組導入 OrderedDict 類
from typing import List, Union  # 從 typing 模組導入 List 和 Union 類型

import numpy as np  # 導入 numpy 模組
import torch  # 導入 torch 模組

__all__ = [  # 定義一個列表，包含所有需要導出的名稱
    "normalize_image",
    "channels_first",
    "scale_intrinsics",
    "pointquaternion_to_homogeneous",
    "poses_to_transforms",
    "create_label_image",
]


def normalize_image(rgb: Union[torch.Tensor, np.ndarray]):  # 定義一個函數，用於將 RGB 圖像的值從 [0, 255] 範圍正規化到 [0, 1] 範圍
    r"""Normalizes RGB image values from :math:`[0, 255]` range to :math:`[0, 1]` range.

    Args:
        rgb (torch.Tensor or numpy.ndarray): RGB image in range :math:`[0, 255]`

    Returns:
        torch.Tensor or numpy.ndarray: Normalized RGB image in range :math:`[0, 1]`

    Shape:
        - rgb: :math:`(*)` (any shape)
        - Output: Same shape as input :math:`(*)`
    """
    if torch.is_tensor(rgb):  # 如果 rgb 是一個張量
        return rgb.float() / 255  # 將 rgb 轉換為浮點數，並除以 255
    elif isinstance(rgb, np.ndarray):  # 如果 rgb 是一個 numpy 陣列
        return rgb.astype(float) / 255  # 將 rgb 轉換為浮點數，並除以 255
    else:
        raise TypeError("Unsupported input rgb type: %r" % type(rgb))  # 拋出異常


def channels_first(rgb: Union[torch.Tensor, np.ndarray]):  # 定義一個函數，用於將圖像從通道最後的表示轉換為通道最先的表示
    r"""Converts from channels last representation :math:`(*, H, W, C)` to channels first representation
    :math:`(*, C, H, W)`

    Args:
        rgb (torch.Tensor or numpy.ndarray): :math:`(*, H, W, C)` ordering `(*, height, width, channels)`

    Returns:
        torch.Tensor or numpy.ndarray: :math:`(*, C, H, W)` ordering

    Shape:
        - rgb: :math:`(*, H, W, C)`
        - Output: :math:`(*, C, H, W)`
    """
    if not (isinstance(rgb, np.ndarray) or torch.is_tensor(rgb)):  # 如果 rgb 不是 numpy 陣列或張量
        raise TypeError("Unsupported input rgb type {}".format(type(rgb)))  # 拋出異常

    if rgb.ndim < 3:  # 如果 rgb 的維度小於 3
        raise ValueError(
            "Input rgb must contain atleast 3 dims, but had {} dims.".format(rgb.ndim)
        )  # 拋出異常
    if rgb.shape[-3] < rgb.shape[-1]:  # 如果 rgb 的通道數大於高度
        msg = "Are you sure that the input is correct? Number of channels exceeds height of image: %r > %r"
        warnings.warn(msg % (rgb.shape[-1], rgb.shape[-3]))  # 發出警告
    ordering = list(range(rgb.ndim))  # 創建一個列表，包含從 0 到 rgb 的維度的所有整數
    ordering[-2], ordering[-1], ordering[-3] = ordering[-3], ordering[-2], ordering[-1]  # 調整列表的順序

    if isinstance(rgb, np.ndarray):  # 如果 rgb 是一個 numpy 陣列
        return np.ascontiguousarray(rgb.transpose(*ordering))  # 將 rgb 轉置，並將其轉換為連續的 numpy 陣列
    elif torch.is_tensor(rgb):  # 如果 rgb 是一個張量
        return rgb.permute(*ordering).contiguous()  # 將 rgb 張量進行維度的重新排列，並將其轉換為連續的張量

"""
這段程式碼定義了兩個函數，scale_intrinsics 和 pointquaternion_to_homogeneous。
scale_intrinsics 函數用於將內參矩陣按照新的框架大小進行縮放，pointquaternion_to_homogeneous 
函數用於將 3D 點和單位四元數轉換為齊次變換

"""
def scale_intrinsics(
    intrinsics: Union[np.ndarray, torch.Tensor],
    h_ratio: Union[float, int],
    w_ratio: Union[float, int],
):  # 定義一個函數，用於將內參矩陣按照新的框架大小進行縮放
    # 函數的實現部分被省略了
    r"""Scales the intrinsics appropriately for resized frames where
    :math:`h_\text{ratio} = h_\text{new} / h_\text{old}` and :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

    Args:
        intrinsics (numpy.ndarray or torch.Tensor): Intrinsics matrix of original frame
        h_ratio (float or int): Ratio of new frame's height to old frame's height
            :math:`h_\text{ratio} = h_\text{new} / h_\text{old}`
        w_ratio (float or int): Ratio of new frame's width to old frame's width
            :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

    Returns:
        numpy.ndarray or torch.Tensor: Intrinsics matrix scaled approprately for new frame size

    Shape:
        - intrinsics: :math:`(*, 3, 3)` or :math:`(*, 4, 4)`
        - Output: Matches `intrinsics` shape, :math:`(*, 3, 3)` or :math:`(*, 4, 4)`

    """
    if isinstance(intrinsics, np.ndarray): # 如果內參矩陣是一個 numpy 陣列
        scaled_intrinsics = intrinsics.astype(np.float32).copy()  # 將內參矩陣轉換為浮點數，並創建一個副本
    elif torch.is_tensor(intrinsics):  # 如果內參矩陣是一個張量
        scaled_intrinsics = intrinsics.to(torch.float).clone() # 將內參矩陣轉換為浮點數，並創建一個副本
    else:
        raise TypeError("Unsupported input intrinsics type {}".format(type(intrinsics)))  # 拋出異常
    if not (intrinsics.shape[-2:] == (3, 3) or intrinsics.shape[-2:] == (4, 4)):  # 如果內參矩陣的形狀不是 (3, 3) 或 (4, 4)
        raise ValueError(
            "intrinsics must have shape (*, 3, 3) or (*, 4, 4), but had shape {} instead".format(
                intrinsics.shape
            )
        )  # 拋出異常
    if (intrinsics[..., -1, -1] != 1).any() or (intrinsics[..., 2, 2] != 1).any():  # 如果內參矩陣的最後一個元素或第三個對角元素不等於 1
        warnings.warn(
            "Incorrect intrinsics: intrinsics[..., -1, -1] and intrinsics[..., 2, 2] should be 1."
        )  # 發出警告

    scaled_intrinsics[..., 0, 0] *= w_ratio  # fx  # 將內參矩陣的 fx 元素乘以寬度比例
    scaled_intrinsics[..., 1, 1] *= h_ratio  # fy  # 將內參矩陣的 fy 元素乘以高度比例
    scaled_intrinsics[..., 0, 2] *= w_ratio  # cx  # 將內參矩陣的 cx 元素乘以寬度比例
    scaled_intrinsics[..., 1, 2] *= h_ratio  # cy  # 將內參矩陣的 cy 元素乘以高度比例
    return scaled_intrinsics  # 返回縮放後的內參矩陣


def pointquaternion_to_homogeneous(
    pointquaternions: Union[np.ndarray, torch.Tensor], eps: float = 1e-12  # 定義一個函數，用於將 3D 點和單位四元數轉換為齊次變換
):
    r"""Converts 3D point and unit quaternions :math:`(t_x, t_y, t_z, q_x, q_y, q_z, q_w)` to
    homogeneous transformations [R | t] where :math:`R` denotes the :math:`(3, 3)` rotation matrix and :math:`T`
    denotes the :math:`(3, 1)` translation matrix:

    .. math::

        \left[\begin{array}{@{}c:c@{}}
        R & T \\ \hdashline
        \begin{array}{@{}ccc@{}}
            0 & 0 & 0
        \end{array}  & 1
        \end{array}\right]

    Args:
        pointquaternions (numpy.ndarray or torch.Tensor): 3D point positions and unit quaternions
            :math:`(tx, ty, tz, qx, qy, qz, qw)` where :math:`(tx, ty, tz)` is the 3D position and
            :math:`(qx, qy, qz, qw)` is the unit quaternion.
        eps (float): Small value, to avoid division by zero. Default: 1e-12

    Returns:
        numpy.ndarray or torch.Tensor: Homogeneous transformation matrices.

    Shape:
        - pointquaternions: :math:`(*, 7)`
        - Output: :math:`(*, 4, 4)`

    """
    if not (
        isinstance(pointquaternions, np.ndarray) or torch.is_tensor(pointquaternions)
    ):  # 如果點和四元數不是 numpy 陣列或張量
        raise TypeError(
            '"pointquaternions" must be of type "np.ndarray" or "torch.Tensor". Got {0}'.format(
                type(pointquaternions)
            )
        )  # 拋出異常
    if not isinstance(eps, float):  # 如果 eps 不是浮點數
        raise TypeError('"eps" must be of type "float". Got {0}.'.format(type(eps)))  # 拋出異常
    if pointquaternions.shape[-1] != 7:  # 如果點和四元數的最後一個維度不等於 7
        raise ValueError(
            '"pointquaternions" must be of shape (*, 7). Got {0}.'.format(
                pointquaternions.shape
            )
        )  # 拋出異常

    output_shape = (*pointquaternions.shape[:-1], 4, 4)  # 計算輸出的形狀
    if isinstance(pointquaternions, np.ndarray):  # 如果點和四元數是一個 numpy 陣列
        t = pointquaternions[..., :3].astype(np.float32)  # 獲取點的座標，並將其轉換為浮點數
        q = pointquaternions[..., 3:7].astype(np.float32) # 獲取四元數，並將其轉換為浮點數
        transform = np.zeros(output_shape, dtype=np.float32)  # 創建一個全為 0 的 numpy 陣列，用於存儲變換矩陣
    else:
        t = pointquaternions[..., :3].float()  # 獲取點的座標，並將其轉換為浮點數
        q = pointquaternions[..., 3:7].float()  # 獲取四元數，並將其轉換為浮點數
        transform = torch.zeros(
            output_shape, dtype=torch.float, device=pointquaternions.device
        )  # 創建一個全為 0 的張量，用於存儲變換矩陣

    q_norm = (0.5 * (q ** 2).sum(-1)[..., None]) ** 0.5   # 計算四元數的範數
    q /= (
        torch.max(q_norm, torch.tensor(eps))
        if torch.is_tensor(q_norm)
        else np.maximum(q_norm, eps)
    )  # 將四元數除以其範數，以確保其為單位四元數

    """
    主要功能是將四元數轉換為變換矩陣。
    首先，它會檢查輸入的四元數是 numpy 陣列還是張量
    ，並使用相應的函數計算四元數的乘積。
    然後，它會從四元數的乘積中提取各個元素，並使用這些元素來計算變換矩陣的元素。最後，它會返回計算出的變換矩陣。
    """

    if isinstance(q, np.ndarray):  # 如果四元數是一個 numpy 陣列
        q = np.matmul(q[..., None], q[..., None, :])  # 計算四元數的乘積
    else:
        q = torch.matmul(q.unsqueeze(-1), q.unsqueeze(-2))  # 如果四元數是一個張量，則使用 torch.matmul 函數計算四元數的乘積
    # 從四元數的乘積中提取各個元素
    txx = q[..., 0, 0]
    tyy = q[..., 1, 1]
    tzz = q[..., 2, 2]
    txy = q[..., 0, 1]
    txz = q[..., 0, 2]
    tyz = q[..., 1, 2]
    twx = q[..., 0, 3]
    twy = q[..., 1, 3]
    twz = q[..., 2, 3]
    # 初始化變換矩陣的對角元素為 1
    transform[..., 0, 0] = 1.0
    transform[..., 1, 1] = 1.0
    transform[..., 2, 2] = 1.0
    transform[..., 3, 3] = 1.0
    # 計算變換矩陣的其他元素
    transform[..., 0, 0] -= tyy + tzz
    transform[..., 0, 1] = txy - twz
    transform[..., 0, 2] = txz + twy
    transform[..., 1, 0] = txy + twz
    transform[..., 1, 1] -= txx + tzz
    transform[..., 1, 2] = tyz - twx
    transform[..., 2, 0] = txz - twy
    transform[..., 2, 1] = tyz + twx
    transform[..., 2, 2] -= txx + tyy
    transform[..., :3, 3] = t

    return transform

"""
這個函數的功能是將姿態轉換為相對於具有身份姿態的第一幀的變換。
它首先複製輸入的姿態，然後對每一幀，如果是第一幀，則將其變換為身份矩陣；
否則，將其與前一幀的逆矩陣相乘，得到相對於第一幀的變換。
"""
def poses_to_transforms(poses: Union[np.ndarray, List[np.ndarray]]):
    r"""Converts poses to transformations w.r.t. the first frame in the sequence having identity pose

    Args:
        poses (numpy.ndarray or list of numpy.ndarray): Sequence of poses in `numpy.ndarray` format.

    Returns:
        numpy.ndarray or list of numpy.ndarray: Sequence of frame to frame transformations where initial
            frame is transformed to have identity pose.

    Shape:
        - poses: Could be `numpy.ndarray` of shape :math:`(N, 4, 4)`, or list of `numpy.ndarray`s of shape
          :math:`(4, 4)`
        - Output: Of same shape as input `poses`
    """
    """
    將姿態轉換為相對於具有身份姿態的第一幀的變換

    Args:
        poses (numpy.ndarray or list of numpy.ndarray): 以 `numpy.ndarray` 格式的姿態序列。

    Returns:
        numpy.ndarray or list of numpy.ndarray: 初始幀被轉換為具有身份姿態的幀到幀變換的序列。

    Shape:
        - poses: 可以是形狀為 :math:`(N, 4, 4)` 的 `numpy.ndarray`，或者是形狀為 :math:`(4, 4)` 的 `numpy.ndarray` 的列表
        - Output: 與輸入 `poses` 的形狀相同
    """
    transformations = copy.deepcopy(poses) # 深度複製 poses，創建一個新的變數 transformations
    for i in range(len(poses)):  # 對 poses 中的每一個元素（即每一幀）
        if i == 0: # 如果是第一幀
            transformations[i] = np.eye(4) # 將變換矩陣設為單位矩陣，因為第一幀相對於自己的變換矩陣是單位矩陣
        else:  # 如果不是第一幀
            transformations[i] = np.linalg.inv(poses[i - 1]).dot(poses[i])  # 計算第 i 幀相對於第 i-1 幀的變換矩陣，並存儲在 transformations 中
    return transformations  # 返回計算出的變換矩陣序列

"""
這個函數的功能是創建標籤圖像。它首先創建一個全零的圖像，
然後對於每個類別，將預測為該類別的像素設為相應的顏色。
最後，返回創建的標籤圖像。這個函數可以用於視覺化網絡的預測結果。
"""
def create_label_image(prediction: np.ndarray, color_palette: OrderedDict):  # 定義一個函數 create_label_image，該函數接受兩個參數：prediction 和 color_palette。
    # prediction 是一個 numpy 陣列，其中每個元素是一個整數，表示一個像素的類別標籤。
    # color_palette 是一個有序字典，其中的每一個鍵值對表示一個類別的索引和對應的 RGB 顏色。
    r"""Creates a label image, given a network prediction (each pixel contains class index) and a color palette.

    Args:
        prediction (numpy.ndarray): Predicted image where each pixel contains an integer,
            corresponding to its class label.
        color_palette (OrderedDict): Contains RGB colors (`uint8`) for each class.

    Returns:
        numpy.ndarray: Label image with the given color palette

    Shape:
        - prediction: :math:`(H, W)`
        - Output: :math:`(H, W)`
    """
    """
    給定網絡預測（每個像素包含類別索引）和色彩調色板，創建標籤圖像。

    Args:
        prediction (numpy.ndarray): 預測圖像，其中每個像素包含一個整數，對應於其類別標籤。
        color_palette (OrderedDict): 包含每個類別的 RGB 顏色 (`uint8`)。

    Returns:
        numpy.ndarray: 帶有給定色彩調色板的標籤圖像

    Shape:
        - prediction: :math:`(H, W)`
        - Output: :math:`(H, W)`
    """
    label_image = np.zeros(
        (prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8
    )    # 創建一個全零的 numpy 陣列 label_image，該陣列的形狀與 prediction 相同，但是有三個通道（對應於 RGB 顏色）。 這個陣列將用於存儲最終的標籤圖像。
    for idx, color in enumerate(color_palette): # 對於 color_palette 中的每一個顏色，獲取其索引（即類別標籤）和 RGB 顏色。
        label_image[prediction == idx] = color  # 將 prediction 中等於當前索引的所有像素的顏色設為當前的 RGB 顏色。
        # 這一步實際上是將預測的類別標籤轉換為對應的 RGB 顏色。
    return label_image   # 返回創建的標籤圖像。
