# 主要功能是分析视频文件，检测并处理黑边，同时提供视频方向适配功能
'''
模块整体功能概述
1.核心功能​​：
检测视频黑边（静态检测和动态检测两种算法）
计算并返回视频裁剪信息（去除黑边）
获取视频基本信息（分辨率、帧率、时长等）
确定最适合的分辨率（用于多视频合并时的兼容性）
2.​​主要用途​​：
视频编辑处理前的预处理
多视频合并时的分辨率适配
黑边自动检测与去除
3.输入输出​​：
​​输入​​：视频文件路径（Path对象）
​​输出​​：VideoInfo对象（包含视频基本信息和裁剪信息）
4.设计特点​​：
支持两种黑边检测算法（静态和动态）
使用采样技术提高处理效率
支持信号机制更新进度条
提供方向适配功能
'''
import random
from collections import Counter
from pathlib import Path
from typing import Tuple

import cv2
import loguru

from src.common.black_remove.img_black_remover import BlackRemover
from src.common.black_remove.video_remover import VideoRemover
from src.config import BlackBorderAlgorithm, cfg  # 58、149
from src.core.datacls import CropInfo, VideoInfo  # 101、106、107
from src.core.enums import Orientation            # 178
from src.signal_bus import SignalBus
from src.utils import get_audio_sample_rate      # 52

black_remover = BlackRemover()
video_remover = VideoRemover()
signal_bus = SignalBus()


def _img_black_remover_start(video_path: Path, sample_rate: float) -> VideoInfo:
    # 使用静态方法检测并去除黑边（基于采样帧分析）
  
    # 重新获取视频信息（帧数、分辨率、FPS）
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width: int = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    audio_sample_rate: int = get_audio_sample_rate(video_path)                      # src.utils待遍历

    # 计算采样帧数（限制最大采样帧数）
    # 如果有黑边则需要获取主体区域坐标(只获取部分百比分帧)
    sample_frames = int(total_frames * sample_rate)
    # 限制最大采样帧数,不然长时间的视频会导致等待时间过长
    max_frames = cfg.get(cfg.video_sample_frame_number)                            # cfg
    sample_frames = min([max_frames, sample_frames])
  
    # 计算每次需要跳过的帧数
    skip_frames = total_frames // sample_frames if sample_frames else 0
    coordinates = []
    if skip_frames <= 0:
        skip_frames = 1

    # 对每帧使用BlackRemover检测黑边并记录坐标
    for i in range(0, total_frames, skip_frames): 
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        # 获取进度条增加的数量
        signal_bus.set_detail_progress_current.emit(i)
        if not ret:
            break
        # Use BlackRemover to get the coordinates of the frame without black borders
        coordinates.append(black_remover.start(img_array=frame)) # 使用BlackRemover.start()获取剪裁黑边后的坐标（(x1,y1,x2,y2)）

    signal_bus.set_detail_progress_finish.emit()
    signal_bus.advance_total_progress.emit(1)
    video.release()

    # 找出最常见的裁剪坐标
    most_common_coordinates = Counter(coordinates).most_common(1)[0][0]

    # 把坐标转化成x, y, w, h
    most_common_coordinates = (
            most_common_coordinates[0],
            most_common_coordinates[1],
            most_common_coordinates[2] - most_common_coordinates[0],
            most_common_coordinates[3] - most_common_coordinates[1]
            )
    # 边界检查确保坐标有效
    x, y, w, h = most_common_coordinates
    x = max(0, x)
    y = max(0, y)
    w = min(width, w)
    h = min(height, h)

    # 如果剪裁区域的宽高和原视频的宽高相同则不剪裁
    if w == width and h == height:
        return VideoInfo(video_path, fps, total_frames, width, height, audio_sample_rate, None)

    loguru.logger.debug(f'[{video_path.name}]的主体区域坐标为{x, y, w, h}')
    signal_bus.set_total_progress_finish.emit()
    signal_bus.set_detail_progress_finish.emit()
    return VideoInfo(video_path, fps, total_frames, width, height, audio_sample_rate,              # src.core.datacls的VideoInfo、CropInfo
                     CropInfo(*most_common_coordinates))


def _video_black_remover_start(video_path: Path) -> VideoInfo:
    # 功能​​：使用动态方法检测并去除黑边（考虑整个视频序列）。

    # 获取视频基本信息
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    audio_sample_rate: int = get_audio_sample_rate(video_path)

    # 获取结果；使用VideoRemover检测黑边（考虑视频动态变化）
    x, y, w, h = video_remover.start(video_path)
    x = max(0, x)
    y = max(0, y)
    w = min(width, w)
    h = min(height, h)
  
    if w == width and h == height:  # 边界检查确保坐标有效
        return VideoInfo(video_path, fps, total_frames, width, height, audio_sample_rate, None)
    return VideoInfo(video_path, fps, total_frames, width, height, audio_sample_rate, CropInfo(x, y, w, h))  # 返回VideoInfo对象（包含裁剪信息）


def get_video_info(video_path: Path, sample_rate: float = 0.5) -> VideoInfo:
    # 功能​​：获取视频信息的主入口函数，根据配置决定是否检测黑边及使用哪种算法。
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    audio_sample_rate: int = get_audio_sample_rate(video_path)

    loguru.logger.debug(f'正在获取视频信息[{video_path.name}]')
    signal_bus.set_detail_progress_reset.emit()
    signal_bus.set_detail_progress_max.emit(total_frames)

    # 是否需要去黑边
    if cfg.get(cfg.video_black_border_algorithm) == BlackBorderAlgorithm.Disable:
        return VideoInfo(video_path, fps, total_frames, width, height, audio_sample_rate, None)

    # 先判断是否有黑边(获取视频中随机的10帧)
    random_frames = random.sample(range(total_frames), 10)
    is_black = False
    for i in random_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            break
        if black_remover.has_black_border(frame):  # 调用BlackRemover()的子方法.has_black_border()检测是否含有黑边
            is_black = True
            break
    if not is_black:
        return VideoInfo(video_path, fps, total_frames, width, height, audio_sample_rate, None)
    video.release()

    # 根据配置选择静态或动态黑边检测算法
    black_remove_algorithm: BlackBorderAlgorithm = cfg.get(cfg.video_black_border_algorithm)
    if black_remove_algorithm == BlackBorderAlgorithm.Static:
        return _img_black_remover_start(video_path, sample_rate)
    elif black_remove_algorithm == BlackBorderAlgorithm.Dynamic:
        return _video_black_remover_start(video_path)
    else:
        raise ValueError(f'未知的黑边去除算法:{black_remove_algorithm}')


def get_most_compatible_resolution(video_info_list: list[VideoInfo],
                                   orientation: Orientation = Orientation.VERTICAL) -> Tuple[int, int]:
    """获取最合适的视频分辨率，功能​​：确定最适合的分辨率（用于多视频合并）"""
    resolutions: list[Tuple[int, int]] = []
    for each in video_info_list:
        width, height = (each.crop.w, each.crop.h) if each.crop else (each.width, each.height)

        # 判断视频的方向,如果视频的方向和用户选择的方向不一致则需要调换宽高
        if (orientation == Orientation.HORIZONTAL and width > height) or (
                orientation == Orientation.VERTICAL and width < height):
            resolutions.append((width, height))
        else:
            resolutions.append((height, width))

    aspect_ratios: list[float] = [i[0] / i[1] for i in resolutions]
    most_common_ratio = Counter(aspect_ratios).most_common(1)[0][0]
    compatible_resolutions = [res for res in resolutions if (res[0] / res[1]) == most_common_ratio]
    compatible_resolutions.sort(key=lambda x: (x[0] * x[1]), reverse=True)
    width, height = compatible_resolutions[0][:2]
    return width, height # 返回最佳分辨率(width, height)


if __name__ == '__main__':
    print(get_video_info(Path(r"E:\load\python\Project\VideoFusion\测试\dy\b7bb97e21600b07f66c21e7932cb7550.mp4"), 0.5))
