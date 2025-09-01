# Project RoboTransfer
# ... 版权和许可证信息 ...
import argparse  # 用于解析命令行参数

import torch  # 主要的深度学习框架
from datasets import load_dataset  # 用于从Hugging Face Hub或本地加载数据集
from PIL import Image  # 用于图像处理
from robotransfer import RoboTransferPipeline  # 核心的扩散模型管道
from robotransfer.utils.image_loading import (  # 项目自定义的工具函数，用于加载图像
    get_dataset_length,
    load_images_from_dataset,
    load_images_from_local,
)
from robotransfer.utils.save_video import save_images_to_mp4  # 用于将图像序列保存为MP4视频

def main():
    # 初始化参数解析器，并设置描述信息
    parser = argparse.ArgumentParser(description="Run RoboTransfer example.")
    # 添加三个主要的命令行参数
    parser.add_argument(
        "--dataset_path",  # 数据集的路径
        type=str,
        default="HorizonRobotics/RoboTransfer-RealData",  # 默认值指向Hugging Face上的数据集
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--refer_image_path",  # 参考图像的路径（用于控制生成内容的外观，如背景纹理、物体材质等）
        type=str,
        default="assets/example_ref_image/gray_grid_desk.png",  # 默认是一张灰色网格桌面的图片
        help="Path to the reference image.",
    )
    parser.add_argument(
        "--output_dir",  # 输出目录的路径，用于保存生成的结果视频
        type=str,
        default="./output",  # 默认输出到当前目录下的output文件夹
        help="Path to save the output video.",
    )
    args = parser.parse_args()  # 解析命令行参数

    # 将解析出的参数赋值给变量，方便后续使用
    dataset_path = args.dataset_path
    refer_image_path = args.refer_image_path
    output_dir = args.output_dir

    # 加载数据集逻辑：判断是从Hugging Face加载还是从本地加载
    if dataset_path.startswith("HorizonRobotics"):  # 如果路径以HorizonRobotics开头
        print(f"Loading dataset from Hugging Face: {dataset_path}")
        dataset = load_dataset(dataset_path)  # 使用datasets库从Hugging Face加载数据集
        load_loacal_dataset = False  # 设置一个标志位，表示不是加载本地数据集
        length = len(dataset["train"])  # 获取数据集中训练集的长度
    else:  # 否则，认为是本地路径
        print(f"Loading local dataset from local path: {dataset_path}")
        load_loacal_dataset = True  # 设置标志位，表示加载本地数据集
        length = get_dataset_length(dataset_path)  # 使用自定义工具函数获取本地数据集的“长度”（可能是帧数或样本数）

    # 加载预训练的RoboTransferPipeline模型
    pipe = RoboTransferPipeline.from_pretrained(
        "HorizonRobotics/RoboTransfer",  # 模型在Hugging Face上的仓库ID
        torch_dtype=torch.bfloat16,  # 指定模型使用的数据类型，bfloat16是一种节省显存且保持数值范围的半精度浮点数
        trust_remote_code=True,  # 信任并执行从远程仓库加载的代码（对于自定义模型管道通常是必需的）
    )
    pipe.to("cuda")  # 将整个管道模型移动到CUDA设备（GPU）上以加速计算

    frames = []  # 初始化一个空列表，用于累积所有生成出的图像帧
    # 开始一个循环，按片段处理数据集（每次处理30帧，步长为30，即不重叠）
    for i in range(0, length - 30, 30):
        # 根据之前设置的标志位，决定从本地还是加载的dataset对象中读取引导图像
        if load_loacal_dataset:
            # 从本地路径加载深度引导图像和法线引导图像
            depth_guider_images, normal_guider_images = load_images_from_local(
                dataset_path, frames_start=i, frames_end=i + 30  # 指定加载的起始帧和结束帧
            )
            save_video = False  # 设置一个标志，可能控制是否在每次循环后保存视频（此处本地加载似乎不立即保存）
        else:
            # 从加载的dataset对象中获取深度引导图像、法线引导图像，并获取save_video标志
            depth_guider_images, normal_guider_images, save_video = (
                load_images_from_dataset(
                    dataset, frames_start=i, frames_end=i + 30
                )
            )

        # 调用管道进行推理生成！这是最核心的一步。
        generated_frames = pipe(  # pipe是RoboTransferPipeline的实例
            image=Image.open(refer_image_path),  # 打开参考图像并传入，控制生成内容的外观
            depth_guider_images=depth_guider_images,  # 传入深度图序列，引导生成内容的几何结构和3D空间关系
            normal_guider_images=normal_guider_images,  # 传入法线图序列，提供表面朝向信息，进一步强化几何一致性
            min_guidance_scale=1.0,  # 分类器自由引导(CFG)的最小尺度，影响生成结果与条件输入的贴合程度
            max_guidance_scale=3,  # CFG的最大尺度
            height=384,  # 生成图像的高度（像素）
            width=640 * 3,  # 生成图像的宽度（像素），640 * 3可能意味着生成三视图或其他宽幅图像
            num_frames=30,  # 要生成的帧数，应与引导图像的帧数匹配
            num_inference_steps=25,  # 扩散过程的推理步数，更多的步数通常可能带来更好的质量但也更耗时
        ).frames[0]  # 获取生成结果的帧（可能是一个列表，取第一个元素）

        frames += generated_frames  # 将本次循环生成的帧添加到总帧列表中

        # 根据save_video标志决定是否保存视频（例如，从Hugging Face加载的数据集可能每个片段都保存）
        if save_video:
            save_images_to_mp4(
                frames, f"{output_dir}/output_frames_final.mp4", fps=10  # 保存为MP4，帧率为10FPS
            )
        # 无论标志如何，每次循环后都保存一个视频（可能会覆盖上一次的）
        save_images_to_mp4(frames, f"{output_dir}/output_frames.mp4", fps=10)


if __name__ == "__main__":
    main()  # 当脚本被直接运行时，调用main函数
