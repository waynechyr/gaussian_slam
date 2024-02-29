"""
Projective geometry utility functions.
"""

from typing import Optional

import torch
from kornia.geometry.linalg import compose_transformations, inverse_transformation

"""
這段程式碼的主要功能是將一組點轉換為齊次座標。
在計算機視覺和圖形學中，齊次座標是一種常用的座標系，它可以方便地表示和處理點和向量的變換，例如旋轉、平移和縮放。
齊次座標的一個重要特性是，它可以用一個額外的維度來表示點和向量的縮放因子，從而使得變換可以用矩陣乘法來表示和計算。這就是為什麼我們需要將點轉換為齊次座標的原因。
在這裡，使用 torch.nn.functional.pad 函數在 pts 的最後一個維度上添加一個常數 1.0，從而將 pts 轉換為齊次座標。
"""
def homogenize_points(pts: torch.Tensor):# 定義一個名為 homogenize_points 的函數，該函數接受一個參數 pts，該參數是一個 torch.Tensor 對象，表示要轉換為齊次座標的點的集合。
    r"""Convert a set of points to homogeneous coordinates.

    Args:
        pts (torch.Tensor): Tensor containing points to be homogenized.

    Shape:
        pts: N x 3 (N-points, and (usually) 3 dimensions)
        (returns): N x 4

    Returns:
        (torch.Tensor): Homogeneous coordinates of pts

    """
    if not isinstance(pts, torch.Tensor):        # 檢查輸入的 pts 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected input type torch.Tensor. Instead got {}".format(type(pts))
        )
    if pts.dim() < 2:# 檢查輸入的 pts 的維度是否小於 2。如果是，則引發 ValueError。
        raise ValueError(
            "Input tensor must have at least 2 dimensions. Got {} instad.".format(
                pts.dim()
            )
        )

    return torch.nn.functional.pad(pts, (0, 1), "constant", 1.0)# 使用 torch.nn.functional.pad 函數將 pts 轉換為齊次座標。該函數在 pts 的最後一個維度上添加一個常數 1.0，從而將 pts 轉換為齊次座標。

"""
主要功能是將一組齊次座標的點轉換為歐幾里得座標。在計算機視覺和圖形學中，齊次座標是一種常用的座標系，它可以方便地表示和處理點和向量的變換，例如旋轉、平移和縮放。
然而，在許多情況下，我們需要將齊次座標轉換為歐幾里得座標，以便於進行計算和視覺化。
這就是我們需要這個函數的原因。在這裡，我們首先從 pts 中提取最後一個座標（即齊次座標的縮放因子），然後計算每個點需要乘以的縮放因子。
如果點在無窮遠處（即縮放因子為 0），則使用縮放因子 1。最後，我們將 pts 中的每個點乘以相應的縮放因子，並且去掉最後一個座標（即縮放因子），從而將 pts 轉換為歐幾里得座標。
"""
def unhomogenize_points(pts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:# 定義一個名為 unhomogenize_points 的函數，該函數接受兩個參數：pts 和 eps。
    # pts 是一個 torch.Tensor 對象，表示要轉換為歐幾里得座標的點的集合。
    # eps 是一個浮點數，用於判斷點是否在無窮遠處。

    r"""Convert a set of points from homogeneous coordinates to Euclidean
    coordinates.

    This is usually done by taking each point (x, y, z, w) and dividing it by
    the last coordinate (w).

    Args:
        pts (torch.Tensor): Tensor containing points to be unhomogenized.

    Shape:
        pts: N x 4 (N-points, and usually 4 dimensions per point)
        (returns): N x 3

    Returns:
        (torch.Tensor): 'Unhomogenized' points

    """
    if not isinstance(pts, torch.Tensor):
        raise TypeError(
            "Expected input type torch.Tensor. Instead got {}".format(type(pts))
        )# 檢查輸入的 pts 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。

    if pts.dim() < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions. Got {} instad.".format(
                pts.dim()
            )
        ) # 檢查輸入的 pts 的維度是否小於 2。如果是，則引發 ValueError。

    # Get points with the last coordinate (scale) as 0 (points at infinity)
    w: torch.Tensor = pts[..., -1:]  # 從 pts 中提取最後一個座標（即齊次座標的縮放因子）。
    # Determine the scale factor each point needs to be multiplied by
    # For points at infinity, use a scale factor of 1 (used by OpenCV
    # and by kornia)
    # https://github.com/opencv/opencv/pull/14411/files
    scale: torch.Tensor = torch.where(torch.abs(w) > eps, 1.0 / w, torch.ones_like(w))# 計算每個點需要乘以的縮放因子。如果點在無窮遠處（即縮放因子為 0），則使用縮放因子 1。

    return scale * pts[..., :-1] # 將 pts 中的每個點乘以相應的縮放因子，並且去掉最後一個座標（即縮放因子），從而將 pts 轉換為歐幾里得座標。


"""
主要功能是將一個四元數轉換為軸角表示法。
在計算機視覺和圖形學中，軸角表示法是一種常用的旋轉表示方法，它由一個旋轉軸和一個旋轉角度組成。
這種表示方法的優點是直觀和易於理解，但是在進行旋轉計算時可能會出現問題，例如旋轉角度的範圍通常被限制在 0 到 360 度之間，這可能會導致某些旋轉無法正確表示。
相比之下，四元數是一種更為強大的旋轉表示方法，它可以表示任意的 3D 旋轉，並且在進行旋轉計算時具有許多優點，例如避免了旋轉角度的範圍限制，並且可以方便地進行旋轉插值。
然而，四元數的理解和使用相對複雜，因此在需要直觀理解和表示旋轉時，我們通常會將四元數轉換為軸角表示法。這就是我們需要這個函數的原因。
在這裡，我們首先從 quat 中提取四元數的各個元素，然後計算旋轉角度和旋轉軸，最後將這些元素組合成軸角表示法。
"""
def quaternion_to_axisangle(quat: torch.Tensor) -> torch.Tensor:# 定義一個名為 quaternion_to_axisangle 的函數，該函數接受一個參數 quat，該參數是一個 torch.Tensor 對象，表示要轉換的四元數。
    r"""Converts a quaternion to an axis angle.

    Args:
        quat (torch.Tensor): Quaternion (qx, qy, qz, qw) (shape:
            :math:`* \times 4`)

    Return:
        axisangle (torch.Tensor): Axis-angle representation. (shape:
            :math:`* \times 3`)

    """
    if not torch.is_tensor(quat): # 檢查輸入的 quat 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected input quat to be of type torch.Tensor."
            " Got {} instead.".format(type(quat))
        )
    if not quat.shape[-1] == 4: # 檢查輸入的 quat 的最後一個維度是否為 4。如果不是，則引發 ValueError。
        raise ValueError(
            "Last dim of input quat must be of shape 4. "
            "Got {} instead.".format(quat.shape[-1])
        )

    # Unpack quat
    qx: torch.Tensor = quat[..., 0]  # 提取四元數的第一個元素
    qy: torch.Tensor = quat[..., 1]  # 提取四元數的第二個元素
    qz: torch.Tensor = quat[..., 2]  # 提取四元數的第三個元素
    sin_sq_theta: torch.Tensor = qx * qx + qy * qy + qz * qz  # 計算旋轉角度的正弦值的平方
    sin_theta: torch.Tensor = torch.sqrt(sin_sq_theta)  # 計算旋轉角度的正弦值
    cos_theta: torch.Tensor = quat[..., 3]  # 提取四元數的第四個元素，即旋轉角度的餘弦值的一半
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    ) # 計算兩倍的旋轉角度

    # 計算每個點需要乘以的縮放因子。如果點在無窮遠處（即縮放因子為 0），則使用縮放因子 1。
    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_sq_theta > 0.0, k_pos, k_neg)  # 根據 sin_sq_theta 的值選擇縮放因子
    # 計算軸角表示法的各個元素
    axisangle: torch.Tensor = torch.zeros_like(quat)[..., :3]  # 創建一個與 quat 相同形狀的零張量
    axisangle[..., 0] = qx * k  # 計算軸角表示法的第一個元素
    axisangle[..., 1] = qy * k  # 計算軸角表示法的第二個元素
    axisangle[..., 2] = qz * k  # 計算軸角表示法的第三個元素

    return axisangle # 返回軸角表示法


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-12):
    # 定義一個函數 normalize_quaternion，該函數接受兩個參數：quaternion 和 eps。
    # quaternion 是一個四元數，eps 是一個很小的數值，用於避免除以零的情況。

    r"""Normalize a quaternion. The quaternion should be in (x, y, z, w)
    format.

    Args:
        quaternion (torch.Tensor): Quaternion to be normalized
            (shape: (*, 4))
        eps (Optional[bool]): Small value, to avoid division by zero
            (default: 1e-12).

    Returns:
        (torch.Tensor): Normalized quaternion (shape: (*, 4))
    """

    if not quaternion.shape[-1] == 4: # 檢查輸入的四元數的最後一個維度是否為 4。如果不是，則引發 ValueError。
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}.".format(quaternion.shape)
        )
    return torch.nn.functional.normalize(quaternion, p=2, dim=-1, eps=eps)
    # 使用 torch.nn.functional.normalize 函數將四元數歸一化，並返回歸一化後的四元數。

"""
這段程式碼的主要功能是將一個四元數轉換為旋轉矩陣。
在計算機視覺和圖形學中，旋轉矩陣是一種常用的旋轉表示方法，它可以方便地表示和處理點和向量的旋轉。
然而，在進行旋轉計算時，我們通常會先將旋轉表示為四元數，然後再將四元數轉換為旋轉矩陣。這就是我們需要這個函數的原因。
在這裡，我們首先將輸入的四元數歸一化，然後將歸一化後的四元數分解為四個分量，然後計算旋轉矩陣的各個元素，最後將這些元素組合成一個 3x3 的矩陣。
"""
def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    # 定義一個函數 quaternion_to_rotation_matrix，該函數接受一個參數 quaternion，該參數是一個四元數。
    r"""Converts a quaternion to a rotation matrix. The quaternion should
    be in (x, y, z, w) format.

    Args:
        quaternion (torch.Tensor): Quaternion to be converted (shape: (*, 4))

    Return:
        (torch.Tensor): Rotation matrix (shape: (*, 3, 3))

    """
    if not quaternion.shape[-1] == 4:
        # 檢查輸入的四元數的最後一個維度是否為 4。如果不是，則引發 ValueError。
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape)
        )

    # Normalize the input quaternion
    quaternion_norm = normalize_quaternion(quaternion)# 使用之前定義的 normalize_quaternion 函數將輸入的四元數歸一化。


    # Unpack the components of the normalized quaternion
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)# 使用 torch.chunk 函數將歸一化後的四元數分解為四個分量：x、y、z 和 w。

    # Compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)
    # 計算旋轉矩陣的各個元素。
    matrix = torch.stack(
        [
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ],
        dim=-1,
    ).view(-1, 3, 3)
    # 使用 torch.stack 函數將計算出的元素堆疊成一個張量，然後使用 view 函數將該張量轉換為 3x3 的矩陣。
    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
        # 如果輸入的四元數只有一個維度，則使用 torch.squeeze 函數去掉矩陣的第一個維度。
    return matrix  # 返回計算出的旋轉矩陣。

"""
主要功能是計算一個 4x4 的 3D 變換矩陣的逆矩陣。在計算機視覺和圖形學中，變換矩陣是一種常用的表示物體在 3D 空間中位置和方向的方式。
然而，在進行變換計算時，我們通常需要計算變換矩陣的逆，以便於進行逆變換。
這就是我們需要這個函數的原因。在這裡，我們首先從 trans 中提取旋轉矩陣和平移向量，然後計算它們的逆，最後將逆旋轉矩陣和逆平移向量組合成逆變換矩陣。
"""
def inverse_transfom_3d(trans: torch.Tensor):
    # 定義一個名為 inverse_transfom_3d 的函數，該函數接受一個參數 trans，該參數是一個 torch.Tensor 對象，表示要求逆的變換矩陣。
    r"""Inverts a 4 x 4 3D transformation matrix.

    Args:
        trans (torch.Tensor): transformation matrix (shape:
            :math:`* \times 4 \times 4`)

    Returns:
        trans_inv (torch.Tensor): inverse of `trans`

    """
    if not torch.is_tensor(trans):
        # 檢查輸入的 trans 的維度是否為 2 或 3，並且最後兩個維度是否為 4x4。如果不是，則引發 ValueError。
        raise TypeError(
            "Expected input trans of type torch.Tensor. Got {} instead.".format(
                type(trans)
            )
        )
    if not trans.dim() in (2, 3) and trans.shape[-2, :] == (4, 4):
        raise ValueError(
            "Input size must be N x 4 x 4 or 4 x 4. Got {} instead.".format(trans.shape)
        )

    # Unpack tensor into rotation and tranlation components
    # 從 trans 中提取旋轉矩陣和平移向量
    rmat: torch.Tensor = trans[..., :3, :3]  # 提取旋轉矩陣
    tvec: torch.Tensor = trans[..., :3, 3]  # 提取平移向量

    # Compute the inverse
    # 計算旋轉矩陣和平移向量的逆
    rmat_inv: torch.Tensor = torch.transpose(rmat, -1, -2)  # 計算旋轉矩陣的逆，即其轉置
    tvec_inv: torch.Tensor = torch.matmul(-rmat_inv, tvec)  # 計算平移向量的逆，即其相反數

    # Pack the inverse rotation and translation into tensor
    # 將逆旋轉矩陣和逆平移向量組合成逆變換矩陣
    trans_inv: torch.Tensor = torch.zeros_like(trans)  # 創建一個與 trans 相同形狀的零張量
    trans_inv[..., :3, :3] = rmat_inv  # 將逆旋轉矩陣放入逆變換矩陣的左上角
    trans_inv[..., :3, 3] = tvec_inv  # 將逆平移向量放入逆變換矩陣的右上角
    trans_inv[..., -1, -1] = 1.0  # 將逆變換矩陣的右下角元素設為 1
     # 返回逆變換矩陣
    return trans_inv

"""
主要功能是組合兩個 4x4 的 3D 變換矩陣。在計算機視覺和圖形學中，變換矩陣是一種常用的表示物體在 3D 空間中位置和方向的方式。
然而，在進行變換計算時，我們通常需要組合多個變換矩陣，以便於進行複雜的變換。這就是我們需要這個函數的原因。
在這裡，我們首先從 trans1 和 trans2 中提取旋轉矩陣和平移向量，然後計算組合後的旋轉矩陣和平移向量，最後將這些元素組合成一個變換矩陣。
"""
def compose_transforms_3d(trans1: torch.Tensor, trans2: torch.Tensor) -> torch.Tensor:
    # 定義一個名為 compose_transforms_3d 的函數，該函數接受兩個參數：trans1 和 trans2，分別表示兩個要組合的變換矩陣。
    r"""Compose two homogeneous 3D transforms.

    Args:
        trans1 (torch.Tensor): first transformation (shape:
            :math:`* \times 4 \times 4`)
        trans2 (torch.Tensor): second transformation (shape:
            :math:`* \times 4 \times 4`)

    Returns:
        trans_cat (torch.Tensor): composed transformation matrix.

    """
    if not torch.is_tensor(trans1):
        # 檢查輸入的 trans1 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected input trans1 of type torch.Tensor. Got {} instead.".format(
                type(trans1)
            )
        )
    if not trans1.dim() in (2, 3) and trans1.shape[-2, :] == (4, 4):
        # 檢查輸入的 trans1 的維度是否為 2 或 3，並且最後兩個維度是否為 4x4。如果不是，則引發 ValueError。
        raise ValueError(
            "Input size must be N x 4 x 4 or 4 x 4. Got {} instead.".format(
                trans1.shape
            )
        )
    if not torch.is_tensor(trans2):
        # 檢查輸入的 trans2 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected input trans2 of type torch.Tensor. Got {} instead.".format(
                type(trans2)
            )
        )
    if not trans2.dim() in (2, 3) and trans2.shape[-2, :] == (4, 4):
        # 檢查輸入的 trans2 的維度是否為 2 或 3，並且最後兩個維度是否為 4x4。如果不是，則引發 ValueError。
        raise ValueError(
            "Input size must be N x 4 x 4 or 4 x 4. Got {} instead.".format(
                trans2.shape
            )
        )
    assert (
        trans1.shape == trans2.shape
    ), "Both input transformations must have the same shape."
    # 檢查兩個輸入變換矩陣的形狀是否相同。如果不同，則引發 AssertionError。

    # Unpack into rmat, tvec
    # 從 trans1 和 trans2 中提取旋轉矩陣和平移向量
    rmat1: torch.Tensor = trans1[..., :3, :3]  # 提取 trans1 的旋轉矩陣
    rmat2: torch.Tensor = trans2[..., :3, :3]  # 提取 trans2 的旋轉矩陣
    tvec1: torch.Tensor = trans1[..., :3, 3]  # 提取 trans1 的平移向量
    tvec2: torch.Tensor = trans2[..., :3, 3]  # 提取 trans2 的平移向量

    # Compute the composition
    # 計算組合後的旋轉矩陣和平移向量
    rmat_cat: torch.Tensor = torch.matmul(rmat1, rmat2) # 計算組合後的旋轉矩陣，即兩個旋轉矩陣的矩陣乘積
    tvec_cat: torch.Tensor = torch.matmul(rmat1, tvec2) + tvec1  # 計算組合後的平移向量，即旋轉後的平移向量加上原始的平移向量

    # Pack into output tensor
    trans_cat: torch.Tensor = torch.zeros_like(trans1)  # 創建一個與 trans1 相同形狀的零張量
    trans_cat[..., :3, :3] = rmat_cat  # 將組合後的旋轉矩陣放入變換矩陣的左上角
    trans_cat[..., :3, 3] = tvec_cat  # 將組合後的平移向量放入變換矩陣的右上角
    trans_cat[..., -1, -1] = 1.0  # 將變換矩陣的右下角元素設為 1
    # 返回組合後的變換矩陣
    return trans_cat

"""
主要功能是將一組點從一個坐標系轉換到另一個坐標系。在計算機視覺和圖形學中，我們經常需要將點從一個坐標系轉換到另一個坐標系，
例如從世界坐標系轉換到相機坐標系，或者從物體坐標系轉換到世界坐標系。
這就是我們需要這個函數的原因。在這裡，我們首先檢查輸入的點和變換矩陣是否符合要求，
然後將點齊次化，接著將變換矩陣應用於點，最後將點從齊次座標轉換為歐幾里得座標。
"""
def transform_pts_3d(pts_b: torch.Tensor, t_ab: torch.Tensor) -> torch.Tensor:
    # 定義一個名為 transform_pts_3d 的函數，該函數接受兩個參數：pts_b 和 t_ab。
    # pts_b 是一個 torch.Tensor 對象，表示要轉換的點的集合。
    # t_ab 是一個 4x4 的變換矩陣，表示從坐標系 b 到坐標系 a 的變換。
    r"""Transforms a set of points `pts_b` from frame `b` to frame `a`, given an SE(3)
    transformation matrix `t_ab`

    Args:
        pts_b (torch.Tensor): points to be transformed (shape: :math:`N \times 3`)
        t_ab (torch.Tensor): homogenous 3D transformation matrix (shape: :math:`4 \times 4`)

    Returns:
        pts_a (torch.Tensor): `pts_b` transformed to the coordinate frame `a`
            (shape: :math:`N \times 3`)

    """
    if not torch.is_tensor(pts_b):
        # 檢查輸入的 pts_b 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected input pts_b of type torch.Tensor. Got {} instead.".format(
                type(pts_b)
            )
        )
    if not torch.is_tensor(t_ab):
        # 檢查輸入的 t_ab 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected input t_ab of type torch.Tensor. Got {} instead.".format(
                type(t_ab)
            )
        )
    if pts_b.dim() < 2:
        # 檢查輸入的 pts_b 的維度是否小於 2。如果是，則引發 ValueError。
        raise ValueError(
            "Expected pts_b to have at least 2 dimensions. Got {} instead.".format(
                pts_b.dim()
            )
        )
    if t_ab.dim() != 2:
         # 檢查輸入的 t_ab 的維度是否為 2。如果不是，則引發 ValueError。
        raise ValueError(
            "Expected t_ab to have 2 dimensions. Got {} instead.".format(t_ab.dim())
        )
    if t_ab.shape[0] != 4 or t_ab.shape[1] != 4:
         # 檢查輸入的 t_ab 的形狀是否為 4x4。如果不是，則引發 ValueError。
        raise ValueError(
            "Expected t_ab to have shape (4, 4). Got {} instead.".format(t_ab.shape)
        )

    # Determine if we need to homogenize the points
    # 判斷是否需要將點齊次化
    if pts_b.shape[-1] == 3:
        pts_b = homogenize_points(pts_b)

    # Apply the transformation
    # 應用變換
    if pts_b.dim() == 4:
        pts_a_homo = torch.matmul(
            t_ab.unsqueeze(0).unsqueeze(0), pts_b.unsqueeze(-1)
        ).squeeze(-1)
    else:
        pts_a_homo = torch.matmul(t_ab.unsqueeze(0), pts_b.unsqueeze(-1))
        # 將變換矩陣應用於點，得到轉換後的點
    pts_a = unhomogenize_points(pts_a_homo)  # 將點從齊次座標轉換為歐幾里得座標

    return pts_a[..., :3]  # 返回轉換後的點

"""
主要功能是將一組點從一個坐標系轉換到另一個坐標系。在計算機視覺和圖形學中，我們經常需要將點從一個坐標系轉換到另一個坐標系，
例如從世界坐標系轉換到相機坐標系，或者從物體坐標系轉換到世界坐標系。這就是我們需要這個函數的原因。
在這裡，我們首先檢查輸入的點和變換矩陣是否符合要求，然後將點齊次化，接著將變換矩陣應用於點，最後將點從齊次座標轉換為歐幾里得座標。
"""
def transform_pts_nd_KF(pts, tform):
    # 定義一個名為 transform_pts_nd_KF 的函數，該函數接受兩個參數：pts 和 tform。
    # pts 是一個 torch.Tensor 對象，表示要轉換的點的集合。
    # tform 是一個變換矩陣，表示從一個坐標系到另一個坐標系的變換。
    r"""Applies a transform to a set of points.

    Args:
        pts (torch.Tensor): Points to be transformed (shape: B x N x D)
            (N points, D dimensions per point; B -> batchsize)
        tform (torch.Tensor): Transformation to be applied
            (shape: B x D+1 x D+1)

    Returns:
        (torch.Tensor): Transformed points (B, N, D)

    """
    if not pts.shape[0] == tform.shape[0]:
        # 檢查輸入的 pts 和 tform 的第一個維度是否相同。如果不同，則引發 ValueError。
        raise ValueError("Input batchsize must be the same for both  tensors")
    if not pts.shape[-1] + 1 == tform.shape[-1]:
        # 檢查輸入的 pts 的最後一個維度加 1 是否等於 tform 的最後一個維度。如果不等，則引發 ValueError。
        raise ValueError(
            "Last input dims must differ by one, i.e., "
            "pts.shape[-1] + 1 should be equal to tform.shape[-1]."
        )

    # Homogenize
    # 齊次化點
    pts_homo = homogenize_points(pts)  # 使用 homogenize_points 函數將點齊次化

    # Transform
    pts_homo_tformed = torch.matmul(tform.unsqueeze(1), pts_homo.unsqueeze(-1))# 使用 torch.matmul 函數將變換矩陣應用於齊次化的點
    pts_homo_tformed = pts_homo_tformed.squeeze(-1)# 使用 squeeze 函數去掉多餘的維度

    # Unhomogenize
    # 將點從齊次座標轉換為歐幾里得座標
    return unhomogenize_points(pts_homo_tformed)  # 使用 unhomogenize_points 函數將點從齊次座標轉換為歐幾里得座標


"""
主要功能是計算兩個 3D 變換矩陣之間的相對變換。在計算機視覺和圖形學中，我們經常需要計算兩個變換之間的相對變換，
例如從一個物體的坐標系到另一個物體的坐標系的變換，或者從一個時間點到另一個時間點的變換。這就是我們需要這個函數的原因。
在這裡，我們首先計算 trans_01 的逆變換矩陣，然後將其與 trans_02 組合，得到從坐標系 ‘1’ 到坐標系 ‘2’ 的相對變換矩陣。
"""
def relative_transform_3d(
    trans_01: torch.Tensor, trans_02: torch.Tensor
) -> torch.Tensor:
      # 定義一個名為 relative_transform_3d 的函數，該函數接受兩個參數：trans_01 和 trans_02。
    # trans_01 和 trans_02 都是 4x4 的 3D 變換矩陣，表示從全局坐標系 '0' 到兩個不同坐標系 '1' 和 '2' 的變換。
    r"""Given two 3D homogeneous transforms `trans_01` and `trans_02`
    in the global frame '0', this function returns a relative
    transform `trans_12`.

    Args:
        trans_01 (torch.Tensor): first transformation (shape:
            :math:`* \times 4 \times 4`)
        trans_02 (torch.Tensor): second transformation (shape:
            :math:`* \times 4 \times 4`)

    Returns:
        trans_12 (torch.Tensor): composed transformation matrix.

    """
    return compose_transforms_3d(inverse_transfom_3d(trans_01), trans_02)
    # 計算 trans_01 的逆變換矩陣，然後將其與 trans_02 組合，得到從坐標系 '1' 到坐標系 '2' 的相對變換矩陣。
    # 這裡使用了兩個輔助函數：inverse_transfom_3d 和 compose_transforms_3d。
    # inverse_transfom_3d 函數用於計算一個變換矩陣的逆變換矩陣。
    # compose_transforms_3d 函數用於組合兩個變換矩陣。
"""
主要功能是計算兩個 3D 變換矩陣之間的相對變換。在計算機視覺和圖形學中，我們經常需要計算兩個變換之間的相對變換，
例如從一個物體的坐標系到另一個物體的坐標系的變換，或者從一個時間點到另一個時間點的變換。
這就是我們需要這個函數的原因。在這裡，我們首先檢查輸入的變換矩陣是否符合要求，然後計算 trans_01 的逆變換矩陣，
接著計算從坐標系 ‘1’ 到坐標系 ‘2’ 的變換矩陣，最後返回這個變換矩陣。
"""
def relative_transformation(
    trans_01: torch.Tensor, trans_02: torch.Tensor, orthogonal_rotations: bool = False
) -> torch.Tensor:
      # 定義一個名為 relative_transformation 的函數，該函數接受兩個參數：trans_01 和 trans_02，分別表示兩個變換矩陣。
    # 另外，還有一個可選參數 orthogonal_rotations，如果設置為 True，則假定 trans_01[:, :3, :3] 是正交旋轉矩陣，這樣可以更高效地計算逆變換。
    r"""Function that computes the relative homogenous transformation from a
    reference transformation :math:`T_1^{0} = \begin{bmatrix} R_1 & t_1 \\
    \mathbf{0} & 1 \end{bmatrix}` to destination :math:`T_2^{0} =
    \begin{bmatrix} R_2 & t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    .. note:: Works with imperfect (non-orthogonal) rotation matrices as well.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \cdot T_0^{2}

    Arguments:
        trans_01 (torch.Tensor): reference transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02 (torch.Tensor): destination transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        orthogonal_rotations (bool): If True, will invert `trans_01` assuming `trans_01[:, :3, :3]` are
            orthogonal rotation matrices (more efficient). Default: False

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: the relative transformation between the transformations.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = gradslam.geometry.geometryutils.relative_transformation(trans_01, trans_02)  # 4x4
    """
    if not torch.is_tensor(trans_01):
        # 檢查輸入的 trans_01 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。

        raise TypeError(
            "Input trans_01 type is not a torch.Tensor. Got {}".format(type(trans_01))
        )
    if not torch.is_tensor(trans_02):
        # 檢查輸入的 trans_02 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Input trans_02 type is not a torch.Tensor. Got {}".format(type(trans_02))
        )
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        # 檢查輸入的 trans_01 的維度是否為 2 或 3，並且最後兩個維度是否為 4x4。如果不是，則引發 ValueError。
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_01.shape)
        )
    if not trans_02.dim() in (2, 3) and trans_02.shape[-2:] == (4, 4):
        # 檢查輸入的 trans_02 的維度是否為 2 或 3，並且最後兩個維度是否為 4x4。如果不是，則引發 ValueError。
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_02.shape)
        )
    if not trans_01.dim() == trans_02.dim():
        # 檢查兩個輸入變換矩陣的維度是否相同。如果不同，則引發 ValueError。
        raise ValueError(
            "Input number of dims must match. Got {} and {}".format(
                trans_01.dim(), trans_02.dim()
            )
        )
    trans_10: torch.Tensor = (
        inverse_transformation(trans_01)
        if orthogonal_rotations
        else torch.inverse(trans_01)
    )
    # 計算 trans_01 的逆變換矩陣。如果 orthogonal_rotations 為 True，則使用 inverse_transformation 函數計算逆變換；否則，使用 torch.inverse 函數計算逆變換。
    trans_12: torch.Tensor = compose_transformations(trans_10, trans_02)
    # 計算從坐標系 '1' 到坐標系 '2' 的變換矩陣，即 trans_10 和 trans_02 的組合。
    return trans_12
    # 返回從坐標系 '1' 到坐標系 '2' 的變換矩陣。
"""
主要功能是將像素座標歸一化，使得每個維度（x，y）現在都在範圍 [-1, 1] 內。在計算機視覺中，我們經常需要將像素座標歸一化，
以便於進行後續的處理和計算。這就是我們需要這個函數的原因。在這裡，我們首先檢查輸入的像素座標是否符合要求，
然後計算每個軸的歸一化因子，最後返回歸一化後的像素座標。
"""
def normalize_pixel_coords(
    pixel_coords: torch.Tensor, height: int, width: int
) -> torch.Tensor:
       # 定義一個名為 normalize_pixel_coords 的函數，該函數接受三個參數：pixel_coords、height 和 width。
    # pixel_coords 是一個 torch.Tensor 對象，表示要歸一化的像素座標。
    # height 和 width 分別表示圖像的高度和寬度。
    r"""Normalizes pixel coordinates, so that each dimension (x, y) is now
    in the range [-1, 1].

    x coordinates get mapped from [0, height-1] to [-1, 1]
    y coordinates get mapped from [0, width-1] to [-1, 1]

    Args:
        pixel_coords (torch.Tensor): pixel coordinates of a grid
            (shape: :math:`* \times 2`)
        height (int): height of the image (x-direction)
        width (int): width of the image (y-direction)

    Returns:
        (torch.Tensor): normalized pixel coordinates (same shape
            as `pixel_coords`.)

    """
    if not torch.is_tensor(pixel_coords):
         # 檢查輸入的 pixel_coords 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected pixel_coords to be of type torch.Tensor. Got {} instead.".format(
                type(pixel_coords)
            )
        )
    if pixel_coords.shape[-1] != 2:
        # 檢查輸入的 pixel_coords 的最後一個維度是否為 2。如果不是，則引發 ValueError。
        raise ValueError(
            "Expected last dimension of pixel_coords to be of size 2. Got {} instead.".format(
                pixel_coords.shape[-1]
            )
        )

    assert type(height) == int, "Height must be an integer."
    assert type(width) == int, "Width must be an integer."
    # 檢查輸入的 height 和 width 是否為整數。如果不是，則引發 AssertionError。


    dtype = pixel_coords.dtype
    device = pixel_coords.device
    # 獲取 pixel_coords 的數據類型和設備。

    height = torch.tensor([height]).type(dtype).to(device)
    width = torch.tensor([width]).type(dtype).to(device)
    # 將 height 和 width 轉換為與 pixel_coords 相同數據類型和設備的張量。

    # Compute normalization factor along each axis
    wh: torch.Tensor = torch.stack([height, width]).type(dtype).to(device)
    # 將 height 和 width 疊加成一個張量。

    norm: torch.Tensor = 2.0 / (wh - 1)
    # 計算每個軸的歸一化因子。

    return norm[:, 0] * pixel_coords - 1
    # 返回歸一化後的像素座標。
"""
這段程式碼的主要功能是將像素座標從範圍 [-1, 1] 反歸一化到 [0, height-1] 和 [0, width-1]。
在計算機視覺中，我們經常需要將像素座標反歸一化，以便於進行後續的處理和計算。這就是我們需要這個函數的原因。
在這裡，我們首先檢查輸入的像素座標是否符合要求，然後計算每個軸的歸一化因子，最後返回反歸一化後的像素座標。
"""
def unnormalize_pixel_coords(
    pixel_coords_norm: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    # 定義一個名為 unnormalize_pixel_coords 的函數，該函數接受三個參數：pixel_coords_norm、height 和 width。
    # pixel_coords_norm 是一個 torch.Tensor 對象，表示要反歸一化的像素座標。
    # height 和 width 分別表示圖像的高度和寬度。

    r"""Unnormalizes pixel coordinates from the range [-1, 1], [-1, 1]
    to [0, `height`-1] and [0, `width`-1] for x and y respectively.

    Args:
        pixel_coords_norm (torch.Tensor): Normalized pixel coordinates
            (shape: :math:`* \times 2`)
        height (int): Height of the image
        width (int): Width of the image

    Returns:
        (torch.Tensor): Unnormalized pixel coordinates

    """
    if not torch.is_tensor(pixel_coords_norm):
        # 檢查輸入的 pixel_coords_norm 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected pixel_coords_norm to be of type torch.Tensor. Got {} instead.".format(
                type(pixel_coords_norm)
            )
        )
    if pixel_coords_norm.shape[-1] != 2:
        # 檢查輸入的 pixel_coords_norm 的最後一個維度是否為 2。如果不是，則引發 ValueError。
        raise ValueError(
            "Expected last dim of pixel_coords_norm to be of shape 2. Got {} instead.".format(
                pixel_coords_norm.shape[-1]
            )
        )

    assert type(height) == int, "Height must be an integer."
    assert type(width) == int, "Width must be an integer."
    # 檢查輸入的 height 和 width 是否為整數。如果不是，則引發 AssertionError。

    dtype = pixel_coords_norm.dtype
    device = pixel_coords_norm.device
    # 獲取 pixel_coords_norm 的數據類型和設備。

    height = torch.tensor([height]).type(dtype).to(device)
    width = torch.tensor([width]).type(dtype).to(device)
    # 將 height 和 width 轉換為與 pixel_coords_norm 相同數據類型和設備的張量。

    # Compute normalization factor along each axis
    wh: torch.Tensor = torch.stack([height, width]).type(dtype).to(device) # 將 height 和 width 疊加成一個張量。
    
    norm: torch.Tensor = 2.0 / (wh - 1) # 計算每個軸的歸一化因子。
    return 1.0 / norm[:, 0] * (pixel_coords_norm + 1) # 返回反歸一化後的像素座標。

"""
主要功能是為一個圖像生成一個座標網格。在計算機視覺中，我們經常需要生成座標網格，以便於進行後續的處理和計算。
這就是我們需要這個函數的原因。在這裡，我們首先根據 normalized_coords 的值生成 xs 和 ys，然後生成座標網格，最後返回結果。
"""
def create_meshgrid(
    height: int, width: int, normalized_coords: Optional[bool] = True
) -> torch.Tensor:
    # 定義一個名為 create_meshgrid 的函數，該函數接受三個參數：height、width 和 normalized_coords。
    # height 和 width 分別表示圖像的高度和寬度。
    # normalized_coords 是一個可選參數，表示是否將座標歸一化到範圍 [-1, 1]。
    r"""Generates a coordinate grid for an image.

    When `normalized_coords` is set to True, the grid is normalized to
    be in the range [-1, 1] (to be consistent with the pytorch function
    `grid_sample`.)

    https://kornia.readthedocs.io/en/latest/utils.html#kornia.utils.create_meshgrid

    Args:
        height (int): Height of the image (number of rows).
        width (int): Width of the image (number of columns).
        normalized_coords (optional, bool): whether or not to
            normalize the coordinates to be in the range [-1, 1].

    Returns:
        (torch.Tensor): grid tensor (shape: :math:`1 \times H \times W \times 2`).

    """

    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coords:
        xs = torch.linspace(-1, 1, height)
        ys = torch.linspace(-1, 1, width)
    else:
        xs = torch.linspace(0, height - 1, height)
        ys = torch.linspace(0, width - 1, width)
    # 根據 normalized_coords 的值，使用 torch.linspace 函數生成 xs 和 ys。
    # 如果 normalized_coords 為 True，則 xs 和 ys 的範圍為 [-1, 1]；否則，xs 的範圍為 [0, height - 1]，ys 的範圍為 [0, width - 1]。
        
    # Generate grid (2 x H x W)
    base_grid: torch.Tensor = torch.stack((torch.meshgrid([xs, ys])))# 使用 torch.meshgrid 函數生成座標網格，然後使用 torch.stack 函數將其疊加成一個張量。
    
    return base_grid.permute(1, 2, 0).unsqueeze(0)  # 1 xH x W x 2
    # 使用 permute 函數將張量的維度進行重排，然後使用 unsqueeze 函數增加一個維度，最後返回結果。
"""
主要功能是將座標從相機框架轉換到像素框架。在計算機視覺中，我們經常需要將座標從一個框架轉換到另一個框架，例如從相機框架轉換到像素框架，或者從世界框架轉換到相機框架。
這就是我們需要這個函數的原因。在這裡，我們首先檢查輸入的座標和投影矩陣是否符合要求，然後將座標轉換到投影矩陣，最後返回包含 u 和 v 座標的張量。
"""
def cam2pixel(
    cam_coords_src: torch.Tensor,
    dst_proj_src: torch.Tensor,
    eps: Optional[float] = 1e-6,
) -> torch.Tensor:
    # 定義一個名為 cam2pixel 的函數，該函數接受三個參數：cam_coords_src、dst_proj_src 和 eps。
    # cam_coords_src 是一個 torch.Tensor 對象，表示在第一個相機框架中定義的像素座標。
    # dst_proj_src 是一個 4x4 的投影矩陣，表示參考和非參考相機框架之間的投影。
    # eps 是一個可選參數，用於防止除以零。
    r"""Transforms coordinates from the camera frame to the pixel frame.

    # based on
    # https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L43

    Args:
        cam_coords_src (torch.Tensor): pixel coordinates (defined in the
            frame of the first camera). (shape: :math:`H \times W \times 3`)
        dst_proj_src (torch.Tensor): projection matrix between the reference
            and the non-reference camera frame. (shape: :math:`4 \times 4`)

    Returns:
        (torch.Tensor): array of [-1, 1] coordinates (shape:
            :math:`H \times W \times 2`)

    """
    assert torch.is_tensor(
        cam_coords_src
    ), "cam_coords_src must be of type torch.Tensor."
    # 檢查輸入的 cam_coords_src 是否為 torch.Tensor 對象。如果不是，則引發 AssertionError。
    assert cam_coords_src.dim() in (3, 4), "cam_coords_src must have 3 or 4 dimensions."
    # 檢查輸入的 cam_coords_src 的維度是否為 3 或 4。如果不是，則引發 AssertionError。
    assert cam_coords_src.shape[-1] == 3
    # 檢查輸入的 cam_coords_src 的最後一個維度是否為 3。如果不是，則引發 AssertionError。
    assert torch.is_tensor(dst_proj_src), "dst_proj_src must be of type torch.Tensor."
    # 檢查輸入的 dst_proj_src 是否為 torch.Tensor 對象。如果不是，則引發 AssertionError。
    assert (
        dst_proj_src.dim() == 2
        and dst_proj_src.shape[0] == 4
        and dst_proj_src.shape[0] == 4
    )
    # 檢查輸入的 dst_proj_src 的維度是否為 2，並且形狀是否為 4x4。如果不是，則引發 AssertionError。

    _, h, w, _ = cam_coords_src.shape # 從 cam_coords_src 的形狀中獲取高度 h 和寬度 w。
    
    pts: torch.Tensor = transform_pts_3d(cam_coords_src, dst_proj_src)# 使用 transform_pts_3d 函數將 cam_coords_src 轉換到 dst_proj_src。

    x: torch.Tensor = pts[..., 0]
    y: torch.Tensor = pts[..., 1]
    z: torch.Tensor = pts[..., 2]
    # 從轉換後的點中提取 x、y 和 z 座標。
    u: torch.Tensor = x / torch.where(z != 0, z, torch.ones_like(z))
    v: torch.Tensor = y / torch.where(z != 0, z, torch.ones_like(z))
    # 計算 u 和 v 座標。如果 z 不為零，則 u 等於 x 除以 z，v 等於 y 除以 z；否則，u 等於 x，v 等於 y。
    return torch.stack([u, v], dim=-1)
    # 返回一個包含 u 和 v 座標的張量。
"""
主要功能是將點從像素框架轉換到相機框架。在計算機視覺中，我們經常需要將點從一個框架轉換到另一個框架，例如從像素框架轉換到相機框架，或者從世界框架轉換到相機框架。
這就是我們需要這個函數的原因。在這裡，我們首先檢查輸入的深度、內參的逆和像素座標是否符合要求，然後將像素座標轉換到內參的逆，最後返回相機座標。
"""
def pixel2cam(
    depth: torch.Tensor, intrinsics_inv: torch.Tensor, pixel_coords: torch.Tensor
) -> torch.Tensor:
    # 定義一個名為 pixel2cam 的函數，該函數接受三個參數：depth、intrinsics_inv 和 pixel_coords。
    # depth 是一個 torch.Tensor 對象，表示源深度圖。
    # intrinsics_inv 是一個 4x4 的矩陣，表示內參的逆。
    # pixel_coords 是一個 torch.Tensor 對象，表示均勻的相機座標網格。

    r"""Transforms points from the pixel frame to the camera frame.

    Args:
        depth (torch.Tensor): the source depth maps (shape:
            :math:`H \times W`)
        intrinsics_inv (torch.Tensor): the inverse of the intrinsics
            (shape: :math:`4 \times 4`)
        pixel_coords (torch.Tensor): the grid of homogeneous camera
            coordinates (shape: :math:`H \times W \times 3`)

    Returns:
        (torch.Tensor): camera coordinates (shape: :math:`H \times W \times 3`)

    """
    if not torch.is_tensor(depth):# 檢查輸入的 depth 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected depth to be of type torch.Tensor. Got {} instead.".format(
                type(depth)
            )
        )
    if not torch.is_tensor(intrinsics_inv):# 檢查輸入的 intrinsics_inv 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected intrinsics_inv to be of type torch.Tensor. Got {} instead.".format(
                type(intrinsics_inv)
            )
        )
    if not torch.is_tensor(pixel_coords):# 檢查輸入的 pixel_coords 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "Expected pixel_coords to be of type torch.Tensor. Got {} instead.".format(
                type(pixel_coords)
            )
        )
    assert (
        intrinsics_inv.shape[0] == 4
        and intrinsics_inv.shape[1] == 4
        and intrinsics_inv.dim() == 2
    )
 # 檢查輸入的 intrinsics_inv 的形狀是否為 4x4，並且維度是否為 2。如果不是，則引發 AssertionError。
    
    cam_coords: torch.Tensor = transform_pts_3d(
        pixel_coords, intrinsics_inv
    )  # .permute(0, 3, 1, 2)
    # 使用 transform_pts_3d 函數將 pixel_coords 轉換到 intrinsics_inv。
    return cam_coords * depth.permute(0, 2, 3, 1)
    # 返回相機座標。這裡，我們將 cam_coords 和 depth 的變換後的結果相乘，得到最終的相機座標。

"""
主要功能是將相機座標投影到圖像上。在計算機視覺中，我們經常需要將相機座標投影到圖像上，以便於進行後續的處理和計算。
這就是我們需要這個函數的原因。在這裡，我們首先檢查輸入的相機座標和投影矩陣是否符合要求，然後將相機座標轉換到投影矩陣，
最後返回包含 u 和 v 座標的張量。
"""
def cam2pixel_KF(
    cam_coords_src: torch.Tensor, P: torch.Tensor, eps: Optional[float] = 1e-6
) -> torch.Tensor:
    # 定義一個名為 cam2pixel_KF 的函數，該函數接受三個參數：cam_coords_src、P 和 eps。
    # cam_coords_src 是一個 torch.Tensor 對象，表示在第一個相機框架中定義的相機座標。
    # P 是一個 4x4 的投影矩陣，表示參考和非參考相機框架之間的投影。
    # eps 是一個可選參數，用於防止除以零。
    r"""Projects camera coordinates onto the image.

    Args:
        cam_coords_src (torch.Tensor): camera coordinates (defined in the
            frame of the first camera). (shape: :math:`H \times W \times 3`)
        P (torch.Tensor): projection matrix between the reference and the
            non-reference camera frame. (shape: :math:`4 \times 4`)

    Returns:
        (torch.Tensor): array of [-1, 1] coordinates (shape:
            :math:`H \times W \times 2`)

    """
    assert torch.is_tensor(
        cam_coords_src
    ), "cam_coords_src must be of type torch.Tensor."
    # 檢查輸入的 cam_coords_src 是否為 torch.Tensor 對象。如果不是，則引發 AssertionError。

    # assert cam_coords_src.dim() > 3, 'cam_coords_src must have > 3 dimensions.'
    assert cam_coords_src.shape[-1] == 3# 檢查輸入的 cam_coords_src 的最後一個維度是否為 3。如果不是，則引發 AssertionError。
    assert torch.is_tensor(P), "dst_proj_src must be of type torch.Tensor."# 檢查輸入的 P 是否為 torch.Tensor 對象。如果不是，則引發 AssertionError。
    assert P.dim() >= 2 and P.shape[-1] == 4 and P.shape[-2] == 4 # 檢查輸入的 P 的維度是否大於等於 2，並且形狀是否為 4x4。如果不是，則引發 AssertionError。

    pts: torch.Tensor = transform_pts_nd_KF(cam_coords_src, P)# 使用 transform_pts_nd_KF 函數將 cam_coords_src 轉換到 P。
    x: torch.Tensor = pts[..., 0]
    y: torch.Tensor = pts[..., 1]
    z: torch.Tensor = pts[..., 2]
     # 從轉換後的點中提取 x、y 和 z 座標。
    u: torch.Tensor = x / torch.where(z != 0, z, torch.ones_like(z))
    v: torch.Tensor = y / torch.where(z != 0, z, torch.ones_like(z))
     # 計算 u 和 v 座標。如果 z 不為零，則 u 等於 x 除以 z，v 等於 y 除以 z；否則，u 等於 x，v 等於 y。

    return torch.stack([u, v], dim=-1)
    # 返回一個包含 u 和 v 座標的張量。

"""
主要功能是對點雲應用剛體變換。在計算機視覺和機器人學中，我們經常需要對點雲進行剛體變換，例如旋轉和平移，以便於進行後續的處理和計算。
這就是我們需要這個函數的原因。在這裡，我們首先檢查輸入的點雲和變換矩陣是否符合要求，然後提取旋轉矩陣和平移向量，
接著將點雲轉置，然後旋轉並平移點雲，最後將轉換後的點雲轉置回原始維度，並返回結果。
"""
def transform_pointcloud(pointcloud: torch.Tensor, transform: torch.Tensor):
    # 定義一個名為 transform_pointcloud 的函數，該函數接受兩個參數：pointcloud 和 transform。
    # pointcloud 是一個 torch.Tensor 對象，表示要轉換的點雲。
    # transform 是一個 4x4 的矩陣，表示 SE(3) 剛體變換矩陣。
    r"""Applies a rigid-body transformation to a pointcloud.

    Args:
        pointcloud (torch.Tensor): Pointcloud to be transformed
                                   (shape: numpts x 3)
        transform (torch.Tensor): An SE(3) rigid-body transform matrix
                                  (shape: 4 x 4)

    Returns:
        transformed_pointcloud (torch.Tensor): Rotated and translated cloud
                                               (shape: numpts x 3)

    """
    if not torch.is_tensor(pointcloud):
        # 檢查輸入的 pointcloud 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "pointcloud should be tensor, but was %r instead" % type(pointcloud)
        )

    if not torch.is_tensor(transform):
        # 檢查輸入的 transform 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "transform should be tensor, but was %r instead" % type(transform)
        )

    if not pointcloud.ndim == 2:
        # 檢查輸入的 pointcloud 的維度是否為 2。如果不是，則引發 ValueError。
        raise ValueError(
            "pointcloud should have ndim of 2, but had {} instead.".format(
                pointcloud.ndim
            )
        )
    if not pointcloud.shape[1] == 3:
        # 檢查輸入的 pointcloud 的形狀是否為 numpts x 3。如果不是，則引發 ValueError。
        raise ValueError(
            "pointcloud.shape[1] should be 3 (x, y, z), but was {} instead.".format(
                pointcloud.shape[1]
            )
        )
    if not transform.shape[-2:] == (4, 4):
        # 檢查輸入的 transform 的形狀是否為 4x4。如果不是，則引發 ValueError。
        raise ValueError(
            "transform should be of shape (4, 4), but was {} instead.".format(
                transform.shape
            )
        )

    # Rotation matrix
    rmat = transform[:3, :3] # 從 transform 中提取旋轉矩陣 rmat。
    # Translation vector
    tvec = transform[:3, 3] # 從 transform 中提取平移向量 tvec。

    # Transpose the pointcloud (to enable broadcast of rotation to each point)
    transposed_pointcloud = torch.transpose(pointcloud, 0, 1) # 將 pointcloud 轉置，以便將旋轉廣播到每個點。
    # Rotate and translate cloud
    transformed_pointcloud = torch.matmul(rmat, transposed_pointcloud) + tvec.unsqueeze(
        1
    )
    # 旋轉並平移點雲。這裡，我們將 rmat 和 transposed_pointcloud 進行矩陣乘法，然後加上 tvec。

    # Transpose the transformed cloud to original dimensions
    transformed_pointcloud = torch.transpose(transformed_pointcloud, 0, 1)
    # 將轉換後的點雲轉置回原始維度。

    return transformed_pointcloud
    # 返回轉換後的點雲。

def transform_normals(normals: torch.Tensor, transform: torch.Tensor):
    # 定義一個名為 transform_normals 的函數，該函數接受兩個參數：normals 和 transform。
    # normals 是一個 torch.Tensor 對象，表示法線向量。
    # transform 是一個 4x4 的矩陣，表示 SE(3) 剛體變換矩陣。
    r"""Applies a rotation to a tensor containing point normals.

    Args:
        normals (torch.Tensor): Normal vectors (shape: numpoints x 3)
    """
    if not torch.is_tensor(normals):
        # 檢查輸入的 normals 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError("normals should be tensor, but was %r instead" % type(normals))

    if not torch.is_tensor(transform):
        # 檢查輸入的 transform 是否為 torch.Tensor 對象。如果不是，則引發 TypeError。
        raise TypeError(
            "transform should be tensor, but was %r instead" % type(transform)
        )

    if not normals.ndim == 2:
        # 檢查輸入的 normals 的維度是否為 2。如果不是，則引發 ValueError。
        raise ValueError(
            "normals should have ndim of 2, but had {} instead.".format(normals.ndim)
        )
    if not normals.shape[1] == 3:
        # 檢查輸入的 normals 的形狀是否為 numpoints x 3。如果不是，則引發 ValueError。
        raise ValueError(
            "normals.shape[1] should be 3 (x, y, z), but was {} instead.".format(
                normals.shape[1]
            )
        )
    if not transform.shape[-2:] == (4, 4):
        # 檢查輸入的 transform 的形狀是否為 4x4。如果不是，則引發 ValueError。
        raise ValueError(
            "transform should be of shape (4, 4), but was {} instead.".format(
                transform.shape
            )
        )

    # Rotation
    R = transform[:3, :3] # 從 transform 中提取旋轉矩陣 R。
    
    # apply transpose to normals
    transposed_normals = torch.transpose(normals, 0, 1)# 將 normals 轉置，以便將旋轉廣播到每個點。


    # transpose after transform
    transformed_normals = torch.transpose(torch.matmul(R, transposed_normals), 0, 1)# 將 R 和 transposed_normals 進行矩陣乘法，然後將結果轉置，得到轉換後的法線。

    return transformed_normals# 返回轉換後的法線。

"""
主要功能是測試前面定義的各種函數，包括創建網格、均質化點、將像素座標轉換為相機座標，以及將相機座標轉換回像素座標。
這些函數在計算機視覺和機器人學中非常常用，因此需要進行測試以確保它們的正確性。
"""
if __name__ == "__main__": # 確保只有在直接運行此腳本時，才會執行下面的程式碼。

    # pts = torch.randn(20, 10, 3)
    # homo = homogenize_points(pts)
    # homo[0:3,0:3,3] = torch.zeros(3)
    # # print(homo)
    # unhomo = unhomogenize_points(homo)
    # # print(unhomo)

    # tf = 2 * torch.eye(4)
    # pts_a = transform_pts_3d(unhomo, tf)
    # # print(pts_a)

    # grid = create_meshgrid(480, 640, False)
    # # # print(grid)
    # grid_norm = normalize_pixel_coords(grid, 480, 640)
    # # # print(grid_norm)
    # # grid_unnorm = unnormalize_pixel_coords(grid_norm, 480, 640)
    # # # print(grid_unnorm)

    # from PinholeCamera import PinholeCamera
    # cam = PinholeCamera.from_params(100, 101, 20, 21, 480, 640)
    # pixels = cam2pixel(pts, cam.extrinsics)
    # depth = torch.randn(20, 10, 1)
    # pxl = torch.randn(20, 10, 3)
    # cam_pts = pixel2cam(depth, cam.intrinsics_inverse(), pxl)
    # print(pixels)

    """
    Testing all functions
    """
    h, w = 32, 32
    f, cx, cy = 5, 16, 16
    # 定義圖像的高度和寬度，以及相機的焦距和主點座標。
    depth_src = torch.ones(1, 1, h, w)
    img_dst = torch.rand(1, 3, h, w)
    # 創建一個深度圖和一個隨機圖像。
    from PinholeCamera import PinholeCamera# 從 PinholeCamera 模塊導入 PinholeCamera 類。

    cam = PinholeCamera.from_params(f, f, cx, cy, h, w, 1.0, 2.0, 3.0) # 使用給定的參數創建一個 PinholeCamera 對象。
    grid = create_meshgrid(h, w, False) # 創建一個網格。
    grid_homo = homogenize_points(grid) # 將網格點均質化。
    px2cm = pixel2cam(depth_src, cam.intrinsics_inverse(), grid_homo) # 將像素座標轉換為相機座標。
    print(px2cm.shape, cam.intrinsics.shape) # 打印轉換後的相機座標的形狀和相機的內參矩陣的形狀。
    cm2px = cam2pixel(px2cm, cam.intrinsics) # 將相機座標轉換回像素座標。
    print(cm2px.shape) # 打印轉換後的像素座標的形狀。