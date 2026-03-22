
import torch
import numpy as np
import os
import warnings

from model_src_ecg.model_code_default import NN_default_parallel

warnings.filterwarnings("ignore")

class ECGPredictor:
    def __init__(self, model_path, device='cpu', num_class=16):
        self.device = torch.device(device)
        self.num_class = num_class
        
        # 结构参数
        self.model = NN_default_parallel(
            nOUT=self.num_class, 
            complexity=64, 
            inputchannel=12, 
            num_layers=14
        )
        
        # 加载正式版权重
        if os.path.exists(model_path):
            print(f" 正在加载正式版最佳模型权重: {os.path.basename(model_path)}")
            state_dict = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.model.load_state_dict(state_dict, strict=True)
        else:
            raise FileNotFoundError(f" 找不到权重文件，请检查路径: {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = [
            "窦性心律 (SR)", "心房颤动 (AFIB)", "心房扑动 (AFL)", 
            "窦性心动过缓 (SB)", "窦性心动过速 (ST)", "房性早搏 (PAC)",
            "室性早搏 (PVC)", "左束支传导阻滞 (LBBB)", "右束支传导阻滞 (RBBB)",
            "一度房室传导阻滞 (1AVB)", "二度房室传导阻滞 (2AVB)", "起搏心律 (PR)",
            "ST段改变 (STC)", "T波改变 (TWC)", "左心室肥大 (LVH)", "正常心电图 (NORM)"
        ]

    def preprocess_data(self, data_path):
        
        try:
            raw_data = np.load(data_path) 
            if raw_data.ndim == 3:
                raw_data = raw_data[0]
        except Exception as e:
            print(f" 未找到有效文件 ({e})，正在使用随机生成的心电数据进行测试...")
            raw_data = np.random.randn(12, 6144)
            
        # 转换为 Tensor
        tensor_data = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0)
        
        # 适配 Conv1d 
        if tensor_data.dim() == 4:
            if tensor_data.shape[2] == 1:
                tensor_data = tensor_data.squeeze(2)
            elif tensor_data.shape[3] == 1:
                tensor_data = tensor_data.squeeze(3)
                
        return tensor_data.to(self.device)

    def predict(self, data_path, threshold=0.5):
        # 1. 数据预处理
        input_tensor = self.preprocess_data(data_path)
        
        # 2. 模型前向传播
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
        # 3. 结果解码
        detected_diseases = []
        for i, prob in enumerate(probs):
            if prob >= threshold:
                detected_diseases.append(f"{self.class_names[i]} (置信度: {prob*100:.1f}%)")
                
        # 4. 组装
        if len(detected_diseases) == 0:
            return "未检测到明显异常，各项心电图特征均未超过疾病阈值。"
        else:
            return "通过 AI 诊断模型检测到以下心电异常特征：\n- " + "\n- ".join(detected_diseases)


# 工程测试入口 
if __name__ == "__main__":
    ckpt_path = r"E:\Train\checkpoint\ECGWFDB_ChapmanShaoxingnosemimediumFTratio0.9seed18HM_BiTCN_fold0_best_checkpoint.pkl"
    
    try:
        predictor = ECGPredictor(model_path=ckpt_path, device='cpu')
        
        print("\n" + "="*50)
        print(" 正在分析数据...")
        
        result_text = predictor.predict("dummy_patient_data.npy")
        
        print("\n【模型诊断结果】：")
        print(result_text)
        print("="*50 + "\n")
        
        print("可以接入 LangChain 了。")
        
    except Exception as e:
        print(f"运行出错：{e}")