import os
import json
import shutil

from modelscope.pipelines import pipeline
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.constant import Tasks

from funasr.datasets.ms_dataset import MsDataset
from funasr.utils.compute_wer import compute_wer


def modelscope_finetune(params):
    if not os.path.exists(params["model_dir"]):
        os.makedirs(params["model_dir"], exist_ok=True)
    
    # 加载数据集
    ds_dict = MsDataset.load(params["dataset_path"])
    
    # 添加 freeze_params 配置
    kwargs = dict(
        model=params["modelscope_model_name"],
        data_dir=ds_dict,
        dataset_type=params["dataset_type"],
        work_dir=params["model_dir"],
        batch_bins=params["batch_bins"],
        max_epoch=params["max_epoch"],
        lr=params["lr"],
        freeze_params=True  # 显式传递 freeze_params 参数
    )
    
    # 构建训练器
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    
    # 验证参数冻结状态
    print("\n冻结参数验证：")
    for name, param in trainer.model.named_parameters():
        if "predictor" not in name and param.requires_grad:
            print(f"发现未冻结参数：{name}")
        elif "predictor" in name and not param.requires_grad:
            print(f"发现错误冻结参数：{name}")
    
    # 开始训练
    trainer.train()
    
    # 复制必要文件
    pretrained_model_path = os.path.join(os.environ["HOME"], ".cache/modelscope/hub", params["modelscope_model_name"])
    required_files = ["am.mvn", "decoding.yaml", "configuration.json"]
    for file_name in required_files:
        shutil.copy(os.path.join(pretrained_model_path, file_name),
                    os.path.join(params["model_dir"], file_name))



def modelscope_infer(params):
    # 准备解码配置
    with open(os.path.join(params["model_dir"], "configuration.json")) as f:
        config_dict = json.load(f)
        config_dict["model"]["am_model_name"] = params["decoding_model_name"]
    
    # 更新模型配置以支持多粒度推理
    config_dict["model"].update({
        "predictor": "MultiScaleCifPredictor",
        "bpe_config": {
            "base_model": "/path/to/base.model",
            "granular_models": {
                "level_1_2": "/path/to/level_1_2.model",
                "level_2_3": "/path/to/level_2_3.model",
                "level_4_5": "/path/to/level_4_5.model"
            }
        },
        "predictor_conf": {
            "idim": 512,
            "l_orders": [1,1,1,1],
            "r_orders": [1,1,1,1],
            "num_scales": 4
        }
    })
    
    with open(os.path.join(params["model_dir"], "configuration.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    # 解码逻辑
    decoding_path = os.path.join(params["model_dir"], "decode_results")
    os.makedirs(decoding_path, exist_ok=True)
    
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model=params["model_dir"],
        output_dir=decoding_path,
        batch_size=64,
        model_revision='v1.0.0'  # 确保使用最新模型配置
    )

    # 数据预处理适配
    audio_in = os.path.join(params["test_data_dir"], "wav.scp")
    inference_pipeline(audio_in=audio_in, 
                      output_bpe_tokens=True,  # 输出多粒度分词结果
                      multi_granularity=True)

    # 结果评估（保持原有逻辑）
    text_in = os.path.join(params["test_data_dir"], "text")
    if os.path.exists(text_in):
        text_proc_file = os.path.join(decoding_path, "1best_recog/token")
        compute_wer(text_in, text_proc_file, os.path.join(decoding_path, "text.cer"))
        os.system("tail -n 3 {}".format(os.path.join(decoding_path, "text.cer")))

if __name__ == '__main__':
    finetune_params = {
    "modelscope_model_name": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "dataset_path": "/path/to/your/dataset",  # 修改为您的数据集路径
    "model_dir": "./checkpoint_custom",
    "dataset_type": "small",  # 如果数据集较大，可以改为 "large"
    "batch_bins": 2000,  # 根据显存大小调整
    "max_epoch": 20,
    "lr": 0.00005,
    "freeze_params": True  # 确保只训练 CIF 部分
}
    
    modelscope_finetune(finetune_params)

    infer_params = {
        "model_dir": "./checkpoint_custom",
        "decoding_model_name": "30epoch.pb",
        "test_data_dir": "./example_data/test/",
        "enable_multi_granularity": True  # 启用多粒度推理
    }
    
    modelscope_infer(infer_params)