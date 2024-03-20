import argparse
import io

import logging
import subprocess
from typing import List
from predict import *
from fastapi import UploadFile, File, HTTPException
from fastapi import FastAPI, BackgroundTasks, Request
from pathlib import Path
from pydub import AudioSegment
from transformers import GPT2LMHeadModel
# 导入音频切割模块
from tools.slice_audio import slice
# 导入声音降噪模块
from UVR5 import uvr
# 导入打标签重采样模块
from labou_train_data import *
# 数据预处理模块
from data_process import *
# 训练模块
from GPT_train import main_GPT
from SoVITS_train import main


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
    in_dir = f"./work_dir/train_audio/{project_id}"
    temp_out_dir = f"./work_dir/train_audio/temp"
    out_dir = f"./work_dir/train_audio/{project_id}"

    # 调用重采样函数
    resample_audio(in_dir, temp_out_dir, out_dir)
    print('数据预处理（打标签与重采样）已经完成')
    # 所有文件处理完毕，返回成功信息
    return {"detail_1": "sucess"}

@app.post("/data_process/{project_id}")
async def data_process(project_id: str):
    preprocess_text_and_extract_features(project_id)
    print('完成文本获取')
    extract_features(project_id)
    print('完成ssl提取')
    process_semantic_embeddings(project_id)
    print('完成语义token提取')
    return {"detail_2": "sucess"}

def train_two_model(project_id):
    main(project_id)
    command = f'/home/www/.conda/envs/py39_paddle/bin/python3 /home/www/GPT-SoVITS/api/GPT_train.py -pro_id {project_id}'
    subprocess.run(command, shell=True, check=True)


@app.post("/train/{project_id}")
async def train(project_id: str,background_tasks: BackgroundTasks):
    background_tasks.add_task(train_two_model, project_id)
    return {"message": "克隆任务已提交"}

@app.get("/check_train/{project_id}")
async def check_train(project_id: str):
    GPT_model_dir = f'/home/www/GPT-SoVITS/work_dir/train/{project_id}/log_s1/ckpt/'
    SoVITS_Model_dir = f'/home/www/GPT-SoVITS/work_dir/train/{project_id}/logs2_weight/'

    gpt_ckpt_exists = any(fname.endswith('.ckpt') for fname in os.listdir(GPT_model_dir) if
                          os.path.isfile(os.path.join(GPT_model_dir, fname)))

    sovits_pth_exists = any(fname.endswith('.pth') for fname in os.listdir(SoVITS_Model_dir) if
                            os.path.isfile(os.path.join(SoVITS_Model_dir, fname)))

    if not gpt_ckpt_exists and not sovits_pth_exists:
        return{"message": "尚未克隆完成，请稍等"}

    model_files = []
    if gpt_ckpt_exists:
        model_files.extend([fname for fname in os.listdir(GPT_model_dir) if fname.endswith('.ckpt')])
    if sovits_pth_exists:
        model_files.extend([fname for fname in os.listdir(SoVITS_Model_dir) if fname.endswith('.pth')])

    return model_files






#
# @app.post("/predict/{project_id}")
# async def predict(project_id: str):
#     # , file: List[UploadFile] = File(...)
#     #
#     # target_directory = f'./work_dir/predict/{project_id}/reference_audio/'
#     #
#     # # 判断音频长度，符合要求上传，不符报错
#     # file_bytes = io.BytesIO(await file.read())
#     # try:
#     #     audio = AudioSegment.from_file(file_bytes)
#     #     audio_length_ms = len(audio)
#     #     audio_length_seconds = audio_length_ms / 1000
#     #     if 3 < audio_length_seconds < 10:
#     #         try:
#     #             file_location = target_directory / file.filename
#     #             with file_location.open("wb") as buffer:
#     #                 shutil.copyfileobj(file.file, buffer)
#     #
#     #         except Exception as e:
#     #             raise HTTPException(status_code=500, detail="错误.请检查文件路径是否正确") from e
#     #         finally:
#     #             file.file.close()
#     #             print(f'文件已经放入指定位置：{target_directory}')
#     #     else:
#     #         raise HTTPException(status_code=400, detail="音频长度错误：上传音频长度应为3S-10S")
#     #
#     # except Exception as e:
#     #     raise HTTPException(status_code=400, detail="无法解析音频文件，请确定格式为MP3或wav")
#
#
#     text_language = "zh"
#     refer_wav_path = "/home/www/GPT-SoVITS/work_dir/hxj_0.wav"
#     prompt_text = '各位老师好，我是来自北京大学人民医院的临床医院大夫'
#     prompt_language = 'zh'
#     target_text = '大家好哦，我是练习时长两年半的工程师王晓楠,喜欢唱，跳 ，篮球'
#
#     # GPT_PATH = f'/home/www/GPT-SoVITS/work_dir/train/{project_id}/log_s1/ckpt/{GPT_name}'
#     # SoVOTS_PATH = f'/home/www/GPT-SoVITS/work_dir/train/{project_id}/logs2_weight/{SoVOTS_name}'
#
#     handle(refer_wav_path, prompt_text, prompt_language, target_text, text_language)
#


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)