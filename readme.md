## requirements依赖
用到了这些库
```bash
    torch
    torchaudio
    librosa
    numpy
```
### 原理
核心是通过梅尔频率倒谱系数（MFCC）和时间调制特征来提取音频信号的特征，并将这两类特征结合在一起，用于音频分类。
最终将这些特征输入到一个神经网络中，进行训练和预测，识别出三种猫的声音，根据数据集分为B, F, I三类。

### get_mfcc
该函数的目的是从给定的音频波形（waveform）中提取梅尔频率倒谱系数（MFCC）特征，并进一步计算其一阶和二阶导数（delta）。这些特征通常用于语音识别或音频分类任务中。
- waveform.dim()：检查输入波形的维度。对于音频数据，waveform 通常是一个一维的向量，表示音频信号的振幅值。
- unsqueeze(0)：如果 waveform 是一维的（形状如 [N]，其中 N 是样本数），则通过 unsqueeze(0) 方法将其转为二维（形状如 [1, N]，表示一个通道的音频信号）。这样做是为了让 torchaudio 能够正确处理波形数据。0表示索引，在最左边添加维度，变成上述的[1,N]
- T.MFCC：这是 torchaudio 提供的梅尔频率倒谱系数（MFCC）提取的类。它将音频信号转换成梅尔频率尺度下的倒谱系数。

### bandpass_filter手动实现滤波器

### get_modulation_features(waveform, sample_rate):
1. 一直报错，后来发现傅里叶变换会产生虚数，使得张量变化产生问题。
2. 修改参数，使用幅度谱magnitude来避免对复数张量进行操作，而后对magnitude进行包络计算，进行带通滤波
3. 通过一半注释定位到时bandpass_filter出现了问题
```python
filtered_waveform = torch.tensor(filtered_waveform.copy(), dtype=torch.float32)
```
- 通过添加copy，避免了负步长产生，解决错误

### if __name__ == '__main__':与terminal区别
1. 如果直接运行python文件，则会执行if __name__ == '__main__'下的代码。
2. 如果在终端中运行python文件，则会执行所有非函数和类的代码。

### get_item
1. 调试过程中不断print形状进行测试，最后调整张量成为合适的维度
2. 保证时间维度相同，在torch.cat的时候
3. 发现label不是张量，而是整数，而后转为长整型张量
4. 后来发现需要每个批次的时间维度也要相同

### dataloader
1. 在调试过程中每次会输出很多批的shape，后来发现一次会处理32批
2. collate_fn=custom_collate_fn 定义函数使得每个批次的时间维度相同

### git命令
1. git add -A 添加所有文件到暂存区
2. git commit -m "提交信息" 提交到本地仓库
3. git push 将本地仓库推送到远程仓库
4. git status 查看当前仓库状态
5. vscode操作以及desktop操作