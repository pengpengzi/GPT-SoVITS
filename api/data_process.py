# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import traceback
from transformers import AutoModelForMaskedLM, AutoTokenizer
from text.cleaner import clean_text
from glob import glob
from tqdm import tqdm
from time import time as ttime
import torch
import librosa
from scipy.io import wavfile
# from feature_extractor import cnhubert
# from my_utils import load_audio
import shutil
import traceback

# 文本获取，封装get_process流程
def preprocess_text_and_extract_features(project_id):
    os.chdir('/home/www/GPT-SoVITS/')
    # inp_wav_dir = f"./work_dir/train_audio/{project_id}"
    inp_text = f"./work_dir/filelists/{project_id}.list"
    exp_name = project_id
    i_part = 0
    all_parts = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    opt_dir = f"./work_dir/text/{project_id}"
    bert_pretrained_dir = './GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/'

    is_half = True
    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)

    opt_dir = os.path.join("./work_dir/1_data_process", exp_name)

    def my_save(fea, path):
        dir = os.path.dirname(path)
        name = os.path.basename(path)
        tmp_path = "{}{}.pth".format(ttime(), i_part)
        torch.save(fea, tmp_path)
        shutil.move(tmp_path, os.path.join(dir, name))

    txt_path = "{}/2-name2text-{}.txt".format(opt_dir, i_part)
    if not os.path.exists(txt_path):
        bert_dir = "{}/3-bert".format(opt_dir)
        os.makedirs(opt_dir, exist_ok=True)
        os.makedirs(bert_dir, exist_ok=True)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
        bert_model.to(device)
        if is_half:
            bert_model = bert_model.half()

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

        def process(data, res):
            for name, text, lan in data:
                try:
                    name = os.path.basename(name)
                    phones, word2ph, norm_text = clean_text(text.replace("%", "-").replace("￥", ","), lan)
                    path_bert = "{}/{}.pt".format(bert_dir, name)
                    if not os.path.exists(path_bert) and lan == "zh":
                        bert_feature = get_bert_feature(norm_text, word2ph)
                        assert bert_feature.shape[-1] == len(phones)
                        my_save(bert_feature, path_bert)
                    phones = " ".join(phones)
                    res.append([name, phones, word2ph, norm_text])
                except:
                    print(name, text, traceback.format_exc())

        todo = []
        res = []
        with open(inp_text, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        language_v1_to_language_v2 = {"ZH": "zh", "zh": "zh", "JP": "ja", "jp": "ja", "JA": "ja", "ja": "ja",
                                      "EN": "en", "en": "en", "En": "en"}
        # for line in lines[int(i_part)::int(all_parts)]:
        for line in lines:
            try:
                wav_name, spk_name, language, text = line.split("|")
                print(wav_name, spk_name, language, text)
                todo.append([wav_name, text, language_v1_to_language_v2.get(language, language)])
            except:
                print(line, traceback.format_exc())

        process(todo, res)
        opt = []
        for name, phones, word2ph, norm_text in res:
            opt.append("{}\t{}\t{}\t{}".format(name, phones, word2ph, norm_text))
        with open(txt_path, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")

    return txt_path


# project_id = 'zsp'
# output_file_path = preprocess_text_and_extract_features(project_id)
# print("Output written to:", output_file_path)

# ssl提取封装
# def extract_features(inp_text, inp_wav_dir, exp_name, i_part, all_parts, _CUDA_VISIBLE_DEVICES, opt_dir,
#                      cnhubert_base_dir, is_half_str="True"):
#     os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES
#     cnhubert.cnhubert_base_path = cnhubert_base_dir
#     is_half = eval(is_half_str)
#
#     hubert_dir = f"{opt_dir}/4-cnhubert"
#     wav32dir = f"{opt_dir}/5-wav32k"
#     os.makedirs(opt_dir, exist_ok=True)
#     os.makedirs(hubert_dir, exist_ok=True)
#     os.makedirs(wav32dir, exist_ok=True)
#
#     maxx = 0.95
#     alpha = 0.5
#     if torch.cuda.is_available():
#         device = "cuda:0"
#     elif torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cpu"
#     model = cnhubert.get_model()
#     if is_half:
#         model = model.half().to(device)
#     else:
#         model = model.to(device)
#
#     nan_fails = []
#
#     def my_save(fea, path):
#         dir = os.path.dirname(path)
#         name = os.path.basename(path)
#         tmp_path = f"{ttime()}{i_part}.pth"
#         torch.save(fea, tmp_path)
#         shutil.move(tmp_path, f"{dir}/{name}")
#
#     def name2go(wav_name, wav_path):
#         hubert_path = f"{hubert_dir}/{wav_name}.pt"
#         if os.path.exists(hubert_path):
#             return
#         # ... [rest of name2go function as it was in the original code] ...
#
#     with open(inp_text, "r", encoding="utf8") as f:
#         lines = f.read().strip("\n").split("\n")
#
#     for line in lines[int(i_part)::int(all_parts)]:
#         try:
#             wav_name, spk_name, language, text = line.split("|")
#             if inp_wav_dir:
#                 wav_name = os.path.basename(wav_name)
#                 wav_path = f"{inp_wav_dir}/{wav_name}"
#             else:
#                 wav_path = wav_name
#                 wav_name = os.path.basename(wav_name)
#             name2go(wav_name, wav_path)
#         except:
#             print(line, traceback.format_exc())
#
#     if len(nan_fails) > 0 and is_half:
#         is_half = False
#         model = model.float()
#         for wav_name in nan_fails:
#             try:
#                 name2go(wav_name)
#             except:
#                 print(wav_name, traceback.format_exc())
#
#
# # Example of how to call the function
# if __name__ == "__main__":
#     # Set these variables before calling the function
#     inp_text = "input_text_file.txt"
#     inp_wav_dir = "/path/to/wav/dir"
#     exp_name = "experiment_name"
#     i_part = "0"
#     all_parts = "1"
#     _CUDA_VISIBLE_DEVICES = "0"
#     opt_dir = "/path/to/output/dir"
#     cnhubert_base_dir = "/path/to/cnhubert/base/dir"
#     is_half_str = "True"
#
#     extract_features(inp_text, inp_wav_dir, exp_name, i_part, all_parts, _CUDA_VISIBLE_DEVICES, opt_dir,
#                      cnhubert_base_dir, is_half_str)