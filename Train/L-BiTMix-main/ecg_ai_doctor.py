
import os
import warnings
import numpy as np
import wfdb
import datetime
import random
from scipy.signal import resample

# LangChain 相关导入
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatTongyi
from ecg_inference import ECGPredictor

warnings.filterwarnings("ignore")

# 界面色彩定义 
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'      # 患者颜色
    GREEN = '\033[92m'     # 成功/系统内部颜色
    YELLOW = '\033[93m'    # 医生回复颜色
    RED = '\033[91m'       # 警告/错误颜色
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

# 配置区
os.environ["DASHSCOPE_API_KEY"] = "sk-f4c4204079f644d988b97c33b8a7d5c9" 

# 模型权重路径
MODEL_PATH = r"E:\Train\checkpoint\ECGWFDB_ChapmanShaoxingnosemimediumFTratio0.9seed18HM_BiTCN_fold4_best_checkpoint.pkl"

# 转换 (WFDB -> NPY) 
def preprocess_wfdb_record(file_path, target_fs=500, target_len=6144):
    """
    内部处理函数：将 WFDB 格式自动转为模型可读的 .npy 格式
    """
    base_path = os.path.splitext(file_path)[0]
    
    try:
        print(f"{Colors.BLUE}\n[系统内部] 检测到 PhysioNet 格式，正在预处理: {os.path.basename(base_path)} ...{Colors.RESET}")
        record = wfdb.rdrecord(base_path)
        signal = record.p_signal.T 
        current_fs = record.fs
        
        if current_fs != target_fs:
            print(f"{Colors.YELLOW}[系统内部] 采样率不匹配，正从 {current_fs}Hz 重采样至 {target_fs}Hz{Colors.RESET}")
            num_samples = int(signal.shape[1] * target_fs / current_fs)
            signal = resample(signal, num_samples, axis=1)
            
        if signal.shape[1] > target_len:
            signal = signal[:, :target_len]
        else:
            pad_width = target_len - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
            
        temp_npy_path = "temp_converted_signal.npy"
        np.save(temp_npy_path, signal.astype(np.float32))
        print(f"{Colors.GREEN}[系统内部] 格式转换完成，已交由 AI 医生分析。{Colors.RESET}")
        return temp_npy_path
    except Exception as e:
        print(f"{Colors.RED}\n[系统内部] 格式转换失败: {e}{Colors.RESET}")
        return None

# 加载诊断引擎与工具
print(f"{Colors.BLUE}正在加载本地心电诊断引擎，请稍候...{Colors.RESET}")
ecg_engine = ECGPredictor(model_path=MODEL_PATH)

def ecg_tool_func(user_input_path):
    """供大模型调用的工具接口"""
    file_path = user_input_path.strip().replace("'", "").replace('"', "")
    
    if file_path.lower().endswith(('.hea', '.mat', '.dat')) or not file_path.lower().endswith('.npy'):
        processed_path = preprocess_wfdb_record(file_path)
        if processed_path:
            return ecg_engine.predict(processed_path)
        else:
            return "文件处理失败，请检查路径是否正确或文件是否完整。"
    
    return ecg_engine.predict(file_path)

tools = [
    Tool(
        name="ECG_Diagnostic_System",
        func=ecg_tool_func,
        description="专业的心电图诊断工具。当用户提供文件路径（如 .npy, .hea 等格式）时，必须调用此工具获取医学标签。输入应为准确的文件路径。"
    )
]

# 初始化 
llm = ChatTongyi(model="qwen-max", temperature=0.4)

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

#System Prompt
system_prompt = """你是一位极具威信、经验丰富且充满人文关怀的心内科急诊主任医师。
你的核心任务是：风险分级、生命优先、严谨解读、持续关怀。请严格按照以下【临床接诊法则】与患者交互：

【一、 致命风险拦截（最高优先级）】
如果患者描述中包含以下“红旗症状”：胸痛/胸闷（尤其是压榨性或持续性）、左肩背放射痛、大汗淋漓、濒死感、黑蒙。
  你的响应策略必须是：
1. 直接打断：如果患者还在讨论打游戏、工作或其他无关紧要的事，必须严厉打断，明确告知其面临【急性心肌梗死甚至心脏骤停（猝死）】的极高风险。
2. 绝对指令：要求患者立刻停止一切活动，平躺或半卧位休息，强烈建议立刻拨打120，绝对禁止自行前往医院。

【二、 关键医疗信息追问（Triage）】
在给出紧急建议后，如果患者尚未提供以下信息，请务必用简短的语言进行追问，以便进一步评估：
1. 症状持续了多久？现在是否还在痛？
2. 以前是否有过高血压、糖尿病、冠心病等病史？
3. 您今年多大年龄？

【三、 心电图严谨解读规则】
调用工具获取心电图标签后：
1. 过滤低置信度：忽略 < 70% 的标签。若互斥（如窦性+起搏），忽略不合理的那个。
2. 强制免责声明：无论模型给出的置信度多高（如 84.3% 或 99%），你都必须向患者强调：“心电图自动分析提示为[xxx]，但由于存在假阳性/假阴性可能，此结果不能作为最终确诊依据，必须交由临床医生复核。”
3. 结合症状定性：如果心电图正常，但患者症状高危，必须强调“心电图正常不能排除急性心梗，症状优先”。

【四、 风险分层与灵活安抚（核心能力）】
你需要具备灵活处理各种人群（儿童、青年、老人）和各种场景的能力，避免让患者过度恐慌：
1. 风险分层与情绪安抚：在给出就医建议前，先对当前症状的风险进行分层解释。例如，面对儿童运动后胸痛或年轻人偶发短暂刺痛，应先安抚：“这种情况需要重视，但大多数与肌肉骨骼或良性因素有关，不必过度恐慌。”
2. 警惕小概率重症：在安抚之后，必须严谨地补充：“不过，由于无法完全排除心肌炎、先天性结构异常等潜在风险，仍建议尽快就医排查。”并在明确诊断前建议暂停剧烈活动。
3. 动态应对：请根据患者具体的年龄、病史和诱因，灵活给出最贴合临床实际的回复，不要死板套用单一模板。

【五、 沟通语气与闭环关怀】
1. 态度：在紧急情况下面对患者的倔强（如坚持打游戏）要表现出“医生的威严与严肃”；在安抚时要表现出“长辈的温暖”。
2. 结尾留门：在对话的最后，一定要留下开放式关怀，例如：“请尽快就医，健康比任何游戏都重要。到了医院有任何检查结果，随时发给我，我帮您解读。”

【六、 差异化结尾】
根据风险等级随机切换结尾，避免模板化：
- 紧急：立刻就医，不要等待！
- 中危：建议近期去医院完成相关检查（如动态心电图、心肌酶等）。
- 低危/咨询：保持观察，若有后续报告可随时找我。

【七、 问诊逻辑与追问机制】
1. 拒绝盲目建议与过早结论：当患者主诉（如心慌、胸痛）信息不足时，严禁直接回复“请就医”。必须先根据患者的描述，灵活追问1-2个关键问题（如：是刺痛还是闷痛？持续了多久？有没有发烧或心慌？既往有没有类似情况？）。
2. 致命拦截：若症状符合急性心梗特征（持续压榨性胸痛、大汗、放射痛），立即打断一切话题，命令其停止活动并拨打120。
"""

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    verbose=True, 
    memory=memory,  
    agent_kwargs={"system_message": system_prompt}, 
    handle_parsing_errors=True
)

# 动态问候 
def get_greeting():
    """根据当前时间生成随机且温馨的问候语"""
    hour = datetime.datetime.now().hour
    
    if 5 <= hour < 12:
        time_str = "早上好"
    elif 12 <= hour < 14:
        time_str = "中午好"
    elif 14 <= hour < 18:
        time_str = "下午好"
    else:
        time_str = "晚上好"
        
    phrases = [
        "请问有什么我可以帮您的吗？您可以直接上传心电图文件交给我分析。",
        "我是您的专属心内科 AI 医生。请问最近有胸闷、心慌的症状吗？需要我帮您解读数据吗？",
        "有什么健康方面的问题想咨询吗？上传您的 .npy 或 .hea 心电文件，我来帮您看看。"
    ]
    
    return f"{Colors.YELLOW}【👨‍⚕️ 医生】:\n{time_str}！{random.choice(phrases)}{Colors.RESET}"


# 多轮循环问诊 
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear') 
    
    print(f"{Colors.HEADER}{Colors.BOLD}\n" + "━"*60)
    print("AI 心内科辅助医生已上线") 
    print("我将协助您解读心电图数据、评估症状风险。")
    print("特性：支持多种心电图格式 | 临床风险识别 | 紧急情况优先提醒 ")
    print("退出请输：exit / quit / 退出")
 
    print("━"*60 + f"{Colors.RESET}\n")
    
    print(get_greeting())
    print(f"{Colors.BLUE}" + "-" * 60 + f"{Colors.RESET}\n")
    
    while True:
        user_input = input(f"{Colors.CYAN}{Colors.BOLD}👤 患者: {Colors.RESET}")
        
        if user_input.lower() in ['exit', 'quit', '退出']:
            print(f"\n{Colors.YELLOW}【👨‍⚕️ 医生】:\n祝您身体健康，再见！{Colors.RESET}\n")
            break
        if not user_input.strip():
            continue
            
        try:
            response = agent.run(input=user_input)
            print(f"\n{Colors.YELLOW}【👨‍⚕️ 医生回复】:\n{response}{Colors.RESET}\n")                   
            print(f"{Colors.RED} 提示：AI结果仅供参考。如有剧烈胸痛、大汗、濒死感，请立即拨打120！{Colors.RESET}")
            print(f"{Colors.BLUE}" + "-" * 60 + f"{Colors.RESET}\n")
            
        except Exception as e:
            print(f"\n{Colors.RED} 系统出错: {e}{Colors.RESET}")
            print("请重试。\n")