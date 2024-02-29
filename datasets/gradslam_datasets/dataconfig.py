import torch  # 導入 torch 模組
import yaml  # 導入 yaml 模組


def load_dataset_config(path, default_path=None):  # 定義一個函數，用於加載數據集配置
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, "r") as f:  # 打開配置文件
        cfg_special = yaml.full_load(f)  # 讀取配置文件

    # check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")  # 獲取繼承的配置文件路徑

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:  # 如果存在繼承的配置文件
        cfg = load_dataset_config(inherit_from, default_path)  # 加載繼承的配置文件
    elif default_path is not None:  # 如果存在默認的配置文件
        with open(default_path, "r") as f: # 打開默認的配置文件
            cfg = yaml.full_load(f)  # 讀取默認的配置文件
    else:
        cfg = dict()  # 創建一個空的字典

    # include main configuration
    update_recursive(cfg, cfg_special)  # 更新配置

    return cfg  # 返回配置


def update_recursive(dict1, dict2):  # 定義一個函數，用於遞歸更新字典
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():  # 遍歷第二個字典的每一個鍵值對
        if k not in dict1:  # 如果第一個字典中不存在該鍵
            dict1[k] = dict()  # 在第一個字典中創建該鍵，並將其值設定為一個空的字典
        if isinstance(v, dict):  # 如果值是一個字典
            update_recursive(dict1[k], v)  # 遞歸更新該鍵的值
        else:
            dict1[k] = v  # 將該鍵的值設定為第二個字典中該鍵的值


def common_dataset_to_batch(dataset):  # 定義一個函數，用於將數據集轉換為批次
    colors, depths, poses = [], [], []  # 初始化顏色、深度和姿態的列表
    intrinsics, embeddings = None, None  # 初始化內參和嵌入的變量
    for idx in range(len(dataset)):  # 遍歷數據集的每一個元素
        _color, _depth, intrinsics, _pose, _embedding = dataset[idx]  # 獲取顏色、深度、內參、姿態和嵌入
        colors.append(_color)  # 將顏色添加到顏色的列表中
        depths.append(_depth)  # 將深度添加到深度的列表中
        poses.append(_pose)  # 將姿態添加到姿態的列表中
        if _embedding is not None:  # 如果嵌入不為 None
            if embeddings is None:  # 如果嵌入的變量為 None
                embeddings = [_embedding]  # 將嵌入的變量設定為一個只包含嵌入的列表
            else:
                embeddings.append(_embedding)  # 將嵌入添加到嵌入的列表中
    colors = torch.stack(colors)  # 將顏色的列表轉換為張量
    depths = torch.stack(depths)  # 將深度的列表轉換為張量
    poses = torch.stack(poses)  # 將姿態的列表轉換為張量
    if embeddings is not None:  # 如果嵌入的列表不為 None
        embeddings = torch.stack(embeddings, dim=1)  # 將嵌入的列表轉換為張量
        # # (1, NUM_IMG, DIM_EMBED, H, W) -> (1, NUM_IMG, H, W, DIM_EMBED)
        # embeddings = embeddings.permute(0, 1, 3, 4, 2)
    colors = colors.unsqueeze(0)  # 為顏色的張量添加一個維度
    depths = depths.unsqueeze(0)  # 為深度的張量添加一個維度
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)  # 為內參的張量添加兩個維度
    poses = poses.unsqueeze(0)  # 為姿態的張量添加一個維度
    colors = colors.float()  # 將顏色的張量的數據類型轉換為浮點數
    depths = depths.float()  # 將深度的張量的數據類型轉換為浮點數
    intrinsics = intrinsics.float()  # 將內參的張量的數據類型轉換為浮點數
    poses = poses.float()  # 將姿態的張量的數據類型轉換為浮點數
    if embeddings is not None:  # 如果嵌入的張量不為 None
        embeddings = embeddings.float()  # 將嵌入的張量的數據類型轉換為浮點數
    return colors, depths, intrinsics, poses, embeddings  # 返回顏色、深度、內參、姿態和嵌入的張量
