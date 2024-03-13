import os
import traceback
import torch
from tools.uvr5.mdxnet import MDXNetDereverb
from tools.uvr5.vr import AudioPre, AudioPreDeEcho

new_working_directory = '/home/www/GPT-SoVITS/'
os.chdir(new_working_directory)



def clear_model_cache(model_name, pre_fun):
    try:
        if model_name == "onnx_dereverb_By_FoxJoy":
            del pre_fun.pred.model
            del pre_fun.pred.model_
        else:
            del pre_fun.model
            del pre_fun
    except Exception as e:
        print(e)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 声音降噪
def uvr(inp_root, save_root_vocal, save_root_ins):
    weight_uvr5_root = "/home/www/GPT-SoVITS/tools/uvr5/uvr5_weights"
    model_names = ['HP5-主旋律人声vocals+其他instrumentals']
    paths = ''
    agg = 10
    format0 = 'wav'
    infos = []
    try:
        for model_name in model_names:
            # 初始化模型
            pre_fun = MDXNetDereverb(15) if model_name == "onnx_dereverb_By_FoxJoy" else \
                (AudioPreDeEcho if "DeEcho" in model_name else AudioPre)(
                    agg=int(agg),
                    model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                    device='cuda',
                    is_half=True
                )

            # 准备文件路径
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)] if inp_root else [path.name for path in paths]

            # 处理每个文件
            for path in paths:
                inp_path = os.path.join(inp_root, path)
                if not os.path.isfile(inp_path):
                    continue

                # 处理音频
                try:
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
                    infos.append(f"{os.path.basename(inp_path)}->{model_name}->成功")
                except Exception as e:
                    infos.append(f"{os.path.basename(inp_path)}->{model_name}->{e}")
                    traceback.print_exc()

            # 清理模型和清空CUDA缓存
            clear_model_cache(model_name, pre_fun)

    except Exception as e:
        infos.append(str(e))
        traceback.print_exc()
    finally:
        # 最后清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for info in infos:
            yield info

# # 输入和输出设置
# inp_root = "/home/www/GPT-SoVITS/work_dir/"
# save_root_vocal = 'work_dir/uvr/干声'
# save_root_ins = 'work_dir/uvr/噪音'
#
#
# # 运行分离过程
# output_info = uvr(inp_root, save_root_vocal, save_root_ins)
# for info in output_info:
#     print(info)
