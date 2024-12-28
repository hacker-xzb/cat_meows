import torch
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from model import CatSoundClassifier
from torchaudio.transforms import Resample
from cat_sound_dataset import get_mfcc, get_modulation_features
from sklearn.metrics import confusion_matrix, classification_report

# 类别映射
LABELS = {0: 'B', 1: 'F', 2:'I'}

def predict_auto(auto_path, model_path="./cat_sound_classifier.pth"):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = CatSoundClassifier(input_channels=296)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 加载和预处理音频
    waveform, sample_rate = librosa.load(auto_path, sr=None)
    waveform = torch.tensor(waveform.copy(), dtype=torch.float32)
    waveform = Resample(orig_freq=sample_rate, new_freq=8000)(waveform)

    # 提取特征
    mfcc_features = get_mfcc(waveform, 8000)
    modulation_features = get_modulation_features(waveform, 8000)

    # 调整维度
    modulation_features = modulation_features.squeeze()
    if modulation_features.dim() == 2:
        modulation_features = modulation_features.unsqueeze(0)

    # 确保时间维度匹配
    if mfcc_features.shape[2] != modulation_features.shape[2]:
        min_length = min(mfcc_features.shape[2], modulation_features.shape[2])
        mfcc_features = mfcc_features[:, :, :min_length]
        modulation_features = modulation_features[:, :, :min_length]

    # 合并特征
    features = torch.cat([mfcc_features, modulation_features], dim=1)
    features = features.squeeze(0)

    # 添加bathc维度
    features = features.unsqueeze(0)

    # 预测
    with torch.no_grad():
        features = features.to(device)
        outputs = model(features)
        _, predicted = outputs.max(1)
        predicted_label = LABELS[predicted.item()]
    
    return predicted_label

if __name__ == '__main__':
    # 收集预测结果
    true_labels = []
    pred_labels = []
    
    # 批量测试文件夹中所有音频
    test_dir = "./dataset"
    correct = 0
    total = 0
    
    # # 测试单个文件
    # auto_path = "./dataset/B_ANI01_MC_FN_SIM01_102.wav"
    # predicted_label = predict_auto(auto_path)
    # print(f'预测结果: {predicted_label}')

    # 批量测试文件夹中所有音频
    # test_dir = "./dataset"
    # correct = 0
    # total = 0

    for file in os.listdir(test_dir):
        if file.endswith('.wav'):
            true_label = file.split('_')[0]
            audio_path = os.path.join(test_dir, file)
            predicted_label = predict_auto(audio_path)

            print(f'文件: {file}, 真实标签: {true_label}, 预测标签: {predicted_label}')

            # 收集结果
            true_labels.append(true_label)
            pred_labels.append(predicted_label)

            if predicted_label == true_label:
                correct += 1
            total += 1

    accuracy = 100 * correct / total
    print(f'\n准确率: {accuracy:.2f}%')

    # 保存预测结果
    prediction_results = {
        'true_labels': true_labels,
        'pred_labels': pred_labels
    }
    with open('prediction_results.json', 'w') as f:
        json.dump(prediction_results, f)