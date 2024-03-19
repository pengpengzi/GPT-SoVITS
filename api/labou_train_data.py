from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import os
import librosa
import soundfile
from multiprocessing import Pool, cpu_count
# from tqdm.notebook import tqdm
from tqdm import tqdm
import shutil

'''
labou1内容 打标签
'''
def check_filelist(list_file_path):
    valid_lines = []
    with open(list_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 拆分行来获取音频路径和文本内容
            parts = line.strip().split('|')
            # 检查是否有四个部分（音频路径、名称、语言代码、文本）
            if len(parts) == 4 and parts[3].strip():
                valid_lines.append(line)
    if valid_lines:
        with open(list_file_path, 'w', encoding='utf-8') as file:
            file.writelines(valid_lines)
    print('已完成.list文件检查')



def get_inference_pipeline(lang_code):
    if lang_code == "ZH":
        return pipeline(
            # task=Tasks.auto_speech_recognition,
            # # 上个版本
            # model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'

            # 带标点的文本打标签
            task=Tasks.auto_speech_recognition,
            model='iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404', model_revision="v2.0.4",
            vad_model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', vad_model_revision="v2.0.4",
            punc_model='iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', punc_model_revision="v2.0.4",

        )


    elif lang_code == "EN":
        return pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_paraformer_asr-en-16k-vocab4199-pytorch')
    elif lang_code == "JP":
        return pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline')
    else:
        raise ValueError("Unsupported language code")


def process_directory(source_dir, character_name, lang_code, start_number, parent_dir_template, project_id):

    if not os.path.exists(source_dir):
        print(f"跳过不存在的文件夹: {source_dir}")
        return start_number

    parent_dir = parent_dir_template.format(character_name=character_name)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    inference_pipeline = get_inference_pipeline(lang_code)
    file_number = start_number

    for dirpath, dirnames, filenames in os.walk(source_dir):
        for file in filenames:
            if file.endswith(".wav"):
                wav_filename = file
                lab_filename = file.replace('.wav', '.lab')
                new_filename_base = f"{character_name}_{file_number}"
                new_lab_file_path = os.path.join(parent_dir, new_filename_base + '.lab')
                new_wav_file_path = os.path.join(parent_dir, new_filename_base + '.wav')

                # 复制.wav文件
                shutil.copy2(os.path.join(dirpath, wav_filename), new_wav_file_path)

                lab_path = os.path.join(dirpath, lab_filename)
                use_recognition = False

                # 检查.lab文件是否存在，及其内容
                if os.path.exists(lab_path):
                    with open(lab_path, 'r', encoding='utf-8') as lab_file:
                        lab_text = lab_file.read().strip()
                        if '{' in lab_text and '}' in lab_text:
                            use_recognition = True
                else:
                    use_recognition = True

                # 根据条件使用语音识别或者.lab文件文本
                if use_recognition:
                    rec_result = inference_pipeline(input=new_wav_file_path)
                    # text = rec_result[0]['text'] # 处理控制的情况
                    if rec_result:
                        text = rec_result[0]['text']
                    else:
                        # 处理空列表的情况
                        text = " "  # 或者设置为你希望的默认值
                else:
                    text = lab_text

                # 检查是否存在{project_id}.list文件，不存在则创建
                directory = "./work_dir/filelists"

                # 创建文件夹（如果它不存在）
                os.makedirs(directory, exist_ok=True)
                output_file = f"{directory}/{project_id}.list"
                line = f"{new_wav_file_path}|{character_name}|{lang_code}|{text}\n"
                if text.strip():
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(line)
                    file_number += 1
                    print(f"Processed: {line}")
                else:
                    print(f"检测到空文本语音：{new_wav_file_path}，不予写入")



    return file_number


'''
labou2内容，音频重采样
'''


def process(item):
    wav_path, temp_out_path, sr = item
    if os.path.exists(wav_path) and wav_path.endswith(".wav"):
        wav, _ = librosa.load(wav_path, sr=sr)
        soundfile.write(temp_out_path, wav, sr)


def resample_audio(in_dir, temp_out_dir, out_dir, sr=44100, processes=0):

    # 创建临时和最终目录
    os.makedirs(temp_out_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1

    pool = Pool(processes=processes)
    tasks = []

    for dirpath, _, filenames in os.walk(in_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                wav_path = os.path.join(dirpath, filename)
                temp_out_path = os.path.join(temp_out_dir, os.path.relpath(wav_path, in_dir))
                os.makedirs(os.path.dirname(temp_out_path), exist_ok=True)
                tasks.append((wav_path, temp_out_path, sr))

    for _ in tqdm(pool.imap_unordered(process, tasks)):
        pass

    pool.close()
    pool.join()

    # 移动文件到最终目录，如果目标文件存在则先删除
    for file in os.listdir(temp_out_dir):
        src_file = os.path.join(temp_out_dir, file)
        dst_file = os.path.join(out_dir, file)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.move(src_file, dst_file)

    # 删除临时目录
    shutil.rmtree(temp_out_dir)
    print("音频重采样完毕并移动到最终目录，临时目录已删除!")





'''
读取配置文件 进行语音识别，为每一段音频打标签
'''
# # 读取配置文件
# with open('process_config.yaml', 'r') as config_file:
#     config = yaml.safe_load(config_file)
#     dir_name = config['项目目录']
#
# # 路径和其他设置
# character_name = f"{dir_name}"
# chinese_dir = f"./work_dir/{dir_name}/audio/temp/"  # 中文文件夹路径
# # english_dir = "./Data/StarRail/audio/temp/bailu_en"  # 英文文件夹路径
# # japanese_dir = "./Data/StarRail/audio/temp/bailu_jp"  # 日语文件夹路径
# parent_dir = f"./work_dir/{dir_name}/audio/wavs/{character_name}"
# output_file = f"./work_dir/{dir_name}/filelists/{dir_name}.list"
#
# # 依次处理中文、英文、日文文件夹
# file_number = process_directory(chinese_dir, character_name, "ZH", 0, parent_dir, output_file)
# # file_number = process_directory(english_dir, character_name, "EN", file_number, parent_dir, output_file)
# # process_directory(japanese_dir, character_name, "JP", file_number, parent_dir, output_file)
#
# print("全部处理完毕!")
#
# '''
# 音频重采样
# '''
# # 设置目录路径
# in_dir = f"./Data/{dir_name}/audio/wavs/{dir_name}/"  # 音频文件的当前位置
# temp_out_dir = f"./Data/{dir_name}/audio/temp/temp"  # 临时存储重采样后的文件
# out_dir = f"./Data/{dir_name}/audio/wavs/{dir_name}/"
#
# # 调用重采样函数
# resample_audio(in_dir, temp_out_dir, out_dir)
# print('数据预处理（打标签与重采样）已经完成')
