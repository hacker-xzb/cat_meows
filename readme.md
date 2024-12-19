## requirements依赖
用到了这些库
```bash
    torch
    torchaudio
    librosa
    numpy
```

### get_mfcc
该函数的目的是从给定的音频波形（waveform）中提取梅尔频率倒谱系数（MFCC）特征，并进一步计算其一阶和二阶导数（delta）。这些特征通常用于语音识别或音频分类任务中。
- waveform.dim()：检查输入波形的维度。对于音频数据，waveform 通常是一个一维的向量，表示音频信号的振幅值。
- unsqueeze(0)：如果 waveform 是一维的（形状如 [N]，其中 N 是样本数），则通过 unsqueeze(0) 方法将其转为二维（形状如 [1, N]，表示一个通道的音频信号）。这样做是为了让 torchaudio 能够正确处理波形数据。0表示索引，在最左边添加维度，变成上述的[1,N]
- T.MFCC：这是 torchaudio 提供的梅尔频率倒谱系数（MFCC）提取的类。它将音频信号转换成梅尔频率尺度下的倒谱系数。

### bandpass_filter手动实现滤波器

### get_modulation_features(waveform, sample_rate):
1. 一直报错，后来发现傅里叶变换会产生虚数，使得张量变化产生问题。
2. 修改参数，使用幅度谱magnitude来避免对复数张量进行操作，而后对magnitude进行包络计算，进行带通滤波

### if __name__ == '__main__':与terminal区别
1. 如果直接运行python文件，则会执行if __name__ == '__main__'下的代码。
2. 如果在终端中运行python文件，则会执行所有非函数和类的代码。