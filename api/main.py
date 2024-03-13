from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from pathlib import Path
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
# 导入音频切割模块
from tools.slice_audio import slice
# 导入声音降噪模块
from UVR5 import uvr
# 导入打标签重采样模块
from labou_train_data import *
# 文本获取模块


app = FastAPI()
def create_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)

@app.post("/upload-audio/{project_id}")
async def upload_audio(project_id: str, files: List[UploadFile] = File(...)):
    base_path = "/home/www/GPT-SoVITS/work_dir"
    directories = ["audio_data", "denoise", "temp", "slice_audio_data"]
    for dir_name in directories:
        create_directory(Path(f"{base_path}/{dir_name}/{project_id}"))

    target_directory = Path(f"{base_path}/audio_data/{project_id}")
    save_root_vocal = Path(f"{base_path}/denoise/{project_id}")
    save_root_ins = Path(f"{base_path}/temp/{project_id}")
    opt_root = Path(f"{base_path}/slice_audio_data/{project_id}")

    for file in files:
        try:
            file_location = target_directory / file.filename
            with file_location.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        except Exception as e:
            raise HTTPException(status_code=500, detail="错误.请检查文件路径是否正确") from e
        finally:
            file.file.close()
            print(f'文件已经放入指定位置：{target_directory}')

    # 对音频进行切分
    inp = target_directory
    threshold = -34 # 音量小于这个值视作静音的备选切割点
    min_length = 4000  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
    min_interval = 300  # 最短切割间隔
    hop_size = 10  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
    max_sil_kept = 500  # 切完后静音最多留多长
    _max = 0.9 # 归一化后最大值多少
    alpha = 0.25
    i_part = 0
    all_part = 1
    slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,i_part,all_part)
    print(f'已完成切分，请检查：{opt_root}')
    # 降噪
    output_info = uvr(opt_root, save_root_vocal, save_root_ins)
    for info in output_info:
        print(info)
    # 打标签+重采样
    character_name = project_id
    chinese_dir = save_root_vocal
    parent_dir = f"./work_dir/train_audio/{project_id}"
    file_number = process_directory(chinese_dir, character_name, "ZH", 0, parent_dir, project_id)

    # 设置重采样目录路径
    in_dir = f"./work_dir/train_audio/{project_id}"  # 音频文件的当前位置
    temp_out_dir = f"./work_dir/train_audio/temp"  # 临时存储重采样后的文件
    out_dir = f"./work_dir/train_audio/{project_id}"

    # 调用重采样函数
    resample_audio(in_dir, temp_out_dir, out_dir)
    print('数据预处理（打标签与重采样）已经完成')
    # 所有文件处理完毕，返回成功信息
    return {"detail": "sucess"}

@app.post("/train/{project_id}")
async def upload_audio(project_id: str):
    inp_wav_dir = f"./work_dir/train_audio/{project_id}"
    inp_text = f"./work_dir//filelists/{project_id}.list"
    exp_name = project_id
    i_part = 0
    all_parts = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    opt_dir = f"./work_dir/text/{project_id}"
    bert_pretrained_dir = 'GPT-SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'
    is_half = 'Ture'
    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
    print(txt_path)
    if os.path.exists(txt_path) == False:
        bert_dir = "%s/3-bert" % (opt_dir)
        os.makedirs(opt_dir, exist_ok=True)
        os.makedirs(bert_dir, exist_ok=True)
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
        if is_half == True:
            bert_model = bert_model.half().to(device)
        else:
            bert_model = bert_model.to(device)

        def get_bert_feature(text, word2ph):
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt")
                for i in inputs:
                    inputs[i] = inputs[i].to(device)
                res = bert_model(**inputs, output_hidden_states=True)
                res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

            assert len(word2ph) == len(text)
            phone_level_feature = []
            for i in range(len(word2ph)):
                repeat_feature = res[i].repeat(word2ph[i], 1)
                phone_level_feature.append(repeat_feature)

            phone_level_feature = torch.cat(phone_level_feature, dim=0)

            return phone_level_feature.T





# 如果你需要运行这个app，请取消下面的注释并安装uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)