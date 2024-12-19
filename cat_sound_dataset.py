import os
print(os.getcwd())
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, Resample
import torchaudio.transforms as T
import torch.nn.functional as F
from scipy.signal import butter, filtfilt

# 定义类别映射
LABELS = {'B': 0, 'F': 1, 'I': 2}  # B: food, F: brushing, I: isolation

# 提取梅尔频率倒谱系数 (MFCC)
def get_mfcc(waveform, sample_rate):
    # 确保输入的波形是2D的
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # 使用torchaudio来提取MFCC
    transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 26, "center": False}
    )
    mfcc = transform(waveform)
    
    # MFCC + 一阶和二阶导数
    delta_mfcc = T.ComputeDeltas()(mfcc)
    delta2_mfcc = T.ComputeDeltas()(delta_mfcc)
    
    # 确保所有特征都是3D的 (channel, feature, time)
    mfcc = mfcc.unsqueeze(1) if mfcc.dim() == 2 else mfcc
    delta_mfcc = delta_mfcc.unsqueeze(1) if delta_mfcc.dim() == 2 else delta_mfcc
    delta2_mfcc = delta2_mfcc.unsqueeze(1) if delta2_mfcc.dim() == 2 else delta2_mfcc

    # 拼接MFCC和它的一阶、二阶导数
    return torch.cat([mfcc, delta_mfcc, delta2_mfcc], dim=1)



# 设计带通滤波器
def bandpass_filter(waveform, sample_rate, low_freq, high_freq):
    """
    设计带通滤波器并应用到波形上
    
    参数:
    waveform: 输入的音频信号 [time_steps] 或 [1, time_steps]
    sample_rate: 采样率
    low_freq: 带通滤波器的低频截止
    high_freq: 带通滤波器的高频截止
    
    返回:
    经带通滤波后的波形
    """
    # 确保 waveform 是一维的 NumPy 数组
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze().cpu().numpy().copy()  # 转为 NumPy 数组并创建副本
    elif isinstance(waveform, np.ndarray):
        waveform = waveform.squeeze().copy()  # 删除多余的维度并创建副本
    else:
        raise TypeError("Unsupported waveform type. Expected torch.Tensor or numpy.ndarray")
    # 计算滤波器参数
    nyquist = sample_rate / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # 设计带通滤波器
    b, a = butter(4, [low, high], btype='band')
    
    # 应用滤波器
    filtered_waveform = filtfilt(b, a, waveform)
    
    # 转换回 torch 张量，确保没有负步长
    filtered_waveform = torch.tensor(filtered_waveform.copy(), dtype=torch.float32)
    
    return filtered_waveform.unsqueeze(0)  # 返回 [1, time_steps] 形状

# 时间调制特征提取函数
def get_modulation_features(waveform, sample_rate):
    """
    提取时间调制特征（例如频率、幅度等的时域变化）
    参数：
    waveform: 输入的音频信号，大小为 [时间步长, 通道数]
    sample_rate: 音频采样率
    
    返回：
    modulation_features: 时间调制特征，大小为 [1, num_features, time_steps]
    """
    # 确保 waveform 是 PyTorch 张量，并且没有负步长
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform.copy())  # 使用 .copy() 创建一个副本，避免负步幅
    elif isinstance(waveform, torch.Tensor):
        waveform = waveform.clone()  # 使用 contiguous() 确保连续内存布局

    # 确保 waveform 是连续的（no negative strides）
    waveform = waveform.contiguous()

    # 确保输入波形为单通道并进行必要的处理
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # 如果是单通道，增加通道维度
    elif waveform.dim() == 2:
        waveform = waveform.mean(dim=0).unsqueeze(0)  # 如果是多通道，取均值并增加通道维度
    
    # 采样率为8kHz
    if sample_rate != 8000:
        raise ValueError("Sample rate should be 8000 Hz as per the paper.")
    
    # 使用短时傅里叶变换（STFT）进行时频分析
    # 设定窗口大小为 512 和 50% 重叠（即 256），对应论文中的 30ms 帧和 10ms 重叠
    n_fft = 512
    hop_length = 256  # 10ms 重叠
    window = torch.hann_window(n_fft)  # 使用 Hann 窗口

    # 计算 STFT
    stft_result = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    
    # 提取幅度谱
    magnitude = stft_result.abs()

    # 计算包络
    # 使用带通滤波器进行包络检测来捕捉信号的幅度变化
    lowpass_filtered = bandpass_filter(magnitude, sample_rate=sample_rate, low_freq=300, high_freq=1000)
    
    # 计算包络的时间变化（差分）
    envelope = lowpass_filtered.diff(dim=-1)  # 计算时间维度上的差分，捕捉变化
    
    # 确保输出的特征维度为 (1, num_features, time_steps)
    modulation_features = envelope.unsqueeze(0)  # 增加 batch 维度
    modulation_features = modulation_features.unsqueeze(0)  # 增加通道维度
    
    return modulation_features

# # 示例：调用该函数
# sample_rate = 8000  # 按论文要求的采样率
# waveform = torch.randn(8000)  # 随便生成一个1秒的信号作为示例

# modulation_features = get_modulation_features(waveform, sample_rate)
# print(modulation_features.shape)  # 应该输出 (1, 1, time_steps)，具体的时间步取决于信号长度和参数设置


class CatSoundDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.files = [f for f in os.listdir(dataset_dir) if f.endswith('.wav')]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
    # 获取音频文件路径
        audio_path = os.path.join(self.dataset_dir, self.files[idx])
        
        # 使用librosa加载音频文件
        waveform, sample_rate = librosa.load(audio_path, sr=None)
        
        # 将其转为Torch张量
        waveform = torch.tensor(waveform.copy(), dtype=torch.float32)
        
        # 对音频进行采样重采样
        waveform = Resample(orig_freq=sample_rate, new_freq=8000)(waveform)
        
        # 提取梅尔频率倒谱系数（MFCC）
        mfcc_features = get_mfcc(waveform, 8000)
        
        # 提取时间调制特征（如果需要）
        modulation_features = get_modulation_features(waveform, 8000)
        
        # # 打印特征的形状以进行调试
        # print(f"MFCC shape: {mfcc_features.shape}")
        # print(f"Modulation features shape: {modulation_features.shape}")
        
        # 调整 modulation_features 的维度
        modulation_features = modulation_features.squeeze()  # 移除所有大小为1的维度
        if modulation_features.dim() == 2:
            modulation_features = modulation_features.unsqueeze(0)  # 添加通道维度
        #print(f"Modulation2 features shape: {modulation_features.shape}")

        # 确保两个特征的时间维度匹配
        if mfcc_features.shape[2] != modulation_features.shape[2]:
            min_length = min(mfcc_features.shape[2], modulation_features.shape[2])
            mfcc_features = mfcc_features[:, :, :min_length]
            modulation_features = modulation_features[:, :, :min_length]

        # 合并特征
        features = torch.cat([mfcc_features, modulation_features], dim=1)
        # print(f"features: {features.shape}")

        # 获取标签
        label = self.get_label(self.files[idx])
        # print(f"label: {label}")
        
        return features, label

    
    def get_label(self, filename):
        """
        从文件名提取标签，假设标签在文件名前缀（B, F, I）
        """
        label_name = filename.split('_')[0]  # 获取文件名前缀（B, F, I）
        return LABELS[label_name]  # 返回对应的标签
    
def custom_collate_fn(batch):
    features, labels = zip(*batch)
    
    # 找到时间维度的最大长度
    max_length = max([f.shape[2] for f in features])

    # 填充每个特征到最大长度
    padded_features = []
    for f in features:
        padding = max_length - f.shape[2]
        padded_f = torch.nn.functional.pad(f, (0, padding))  # 填充
        padded_features.append(padded_f)
    
    # 将所有样本堆叠成一个批次
    features_batch = torch.stack(padded_features, 0)
    
    # 将标签转换为张量，假设标签是整数，可以直接转换为长整型张量
    labels_batch = torch.tensor(labels, dtype=torch.long)

    return features_batch, labels_batch



# 数据加载器示例
dataset_dir = './dataset'  # 数据集路径
dataset = CatSoundDataset(dataset_dir)

# for features, label in dataset:
#     print(features.shape, label)
#     break

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

for batch_idx, (features, labels) in enumerate(dataloader):
    # 处理每个批次的 features 和 labels
    print(f"Batch {batch_idx} - Features shape: {features.shape}, Labels shape: {labels.shape}")

# def test_dataset():
#     dataset = CatSoundDataset('./dataset')  # 使用您的数据集路径
#     for i in range(min(5, len(dataset))):  # 测试前5个样本或所有样本（如果少于5个）
#         print(f"\nTesting sample {i}")
#         features, label = dataset[i]
#         print(f"Features shape: {features.shape}, Label: {label}")

# if __name__ == "__main__":
#     test_dataset()
