# 这段代码实现了一个视频处理程序的协调器类 ProgramCoordinator，负责管理整个视频处理流程，包括视频分析、黑边处理、分辨率调整、旋转和合并等操作。
'''
1.核心功能​​：
协调多个视频文件的处理流程
自动检测并去除视频黑边
统一调整视频分辨率*
支持视频旋转操作*
可选合并多个视频为一个*
提供断点续做功能（任务恢复）
实时进度更新和信号通知
2.​​输入输出​​：
​​输入​​：
input_video_path_list: 待处理的视频文件路径列表
orientation: 视频方向（水平或垂直）
rotation: 旋转角度
​​输出​​：
处理后的视频文件（或合并后的视频）
返回输出目录路径
3.流程
初始化 → 任务恢复设置 → 视频分析 → 分辨率确定 → 逐个处理视频 → 可选合并 → 输出结果
'''
import os
import time
from collections import Counter
from pathlib import Path

import loguru

from src.common.black_remove_algorithm.black_remove_algorithm import BlackRemoveAlgorithm
from src.common.black_remove_algorithm.img_black_remover import IMGBlackRemover
from src.common.black_remove_algorithm.video_remover import VideoRemover
from src.common.processors.processor_global_var import ProcessorGlobalVar # 全局变量处理器
from src.common.task_resumer.task_resumer import TaskResumer 
from src.common.task_resumer.task_resumer_manager import TaskResumerManager # 任务恢复管理器
from src.common.video_handler import VideoHandler # 视频处理核心类*
from src.common.video_info_reader import VideoInfoReader
from src.config import BlackBorderAlgorithm, VideoProcessEngine, VideoResolution, cfg # *
from src.core.datacls import VideoInfo
from src.core.enums import Orientation, Rotation
from src.signal_bus import SignalBus
from src.utils import move_file_to_output_dir


class ProgramCoordinator:
    def __init__(self):
        self._start_time = time.time()

        self._signal_bus = SignalBus() # 信号总线，用于进度更新通知
        self._task_resumer_manager = TaskResumerManager() # 任务恢复管理器，支持断点续做
        self._processor_global_var = ProcessorGlobalVar() # 全局变量处理器，存储处理过程中的共享数据
        self._video_handler = VideoHandler() # 视频处理器，执行实际视频处理操作

    @loguru.logger.catch(reraise=True)
    def process(self, input_video_path_list: list[Path], orientation: Orientation, rotation: Rotation) -> Path | None:
        # 输入：input_video_path_list: 待处理视频路径列表；orientation: 视频方向（横屏/竖屏）；rotation: 旋转角度
        # 输出：返回值​​: 输出目录路径
        # sourcery skip: low-code-quality
        # 清空全局变量，存储方向和旋转信息
        self._processor_global_var.clear() 
        self._processor_global_var.update('orientation', orientation)
        self._processor_global_var.update('rotation_angle', rotation.value)
        self._task_resumer_manager.clear()
        loguru.logger.debug(self._processor_global_var)

        self._signal_bus.set_detail_progress_reset.emit()
        self._signal_bus.set_total_progress_reset.emit()
        self._signal_bus.set_total_progress_max.emit(len(input_video_path_list))
        self._signal_bus.set_total_progress_description.emit("分析视频")

        # 将所有任务设置为未完成。为每个视频创建任务记录器，保存任务状态（支持断点续处理）
        for each_resumer in input_video_path_list:
            task_resumer = TaskResumer(each_resumer)
            self._task_resumer_manager.append_task(task_resumer)
        self._task_resumer_manager.save()

        # 读取视频信息，
        black_remove_algorithm_impl = self._get_black_remove_algorithm()
        video_info_list: list[VideoInfo] = []
        finished_video_path_list: list[Path] = []
        for each_path in input_video_path_list:
            video_info = VideoInfoReader(each_path).get_video_info(black_remove_algorithm_impl) # 返回VideoInfo格式数据
            loguru.logger.debug(video_info)
            video_info_list.append(video_info)

            self._signal_bus.advance_total_progress.emit(1)

        target_width, target_height = self._get_video_resolution(video_info_list, orientation) # 根据配置计算最佳输出分辨率
        # 存储到全局变量
        self._processor_global_var.get_data()['target_width'] = target_width
        self._processor_global_var.get_data()['target_height'] = target_height
        self._signal_bus.set_total_progress_finish.emit()
        self._signal_bus.set_detail_progress_finish.emit()
        loguru.logger.debug(f'任务恢复器保存完成,任务数:{len(self._task_resumer_manager.task_list)}')

        self._signal_bus.set_total_progress_reset.emit()
        self._signal_bus.set_detail_progress_reset.emit()
        self._signal_bus.set_total_progress_description.emit("处理视频")
        self._signal_bus.set_total_progress_max.emit(len(video_info_list))

        # 逐个处理未完成视频
        is_merge: bool = cfg.get(cfg.merge_video) # *
        for index, each_resumer in enumerate(self._task_resumer_manager.uncompleted_task_list):
            video_info: VideoInfo = video_info_list[index]
            output_file_path = self._process_single_video(each_resumer, finished_video_path_list, video_info) # 调用_process_single_video处理单个视频
            if not is_merge:
                move_file_to_output_dir([output_file_path])
                loguru.logger.debug(f'已完成视频{output_file_path}的处理,已移动到输出目录')

        if is_merge: # 可选视频合并功能
            finished_video_path = self._video_handler.merge_videos(finished_video_path_list)
            finished_video_path_list.clear()
            finished_video_path_list.append(finished_video_path)

            move_file_to_output_dir(finished_video_path_list) # 移动文件到输出目录

        output_dir: Path = Path(cfg.get(cfg.output_dir))
        os.startfile(output_dir) # 自动打开输出文件夹(windows) # print(f"处理完成！输出文件在：{output_dir}") 

        self._signal_bus.set_total_progress_finish.emit()
        self._signal_bus.set_detail_progress_finish.emit()
        self._signal_bus.finished.emit()
        self._signal_bus.finished.emit()

        loguru.logger.info(
            f'程序执行完成一共处理{len(input_video_path_list)}个视频,耗时: {time.time() - self._start_time}秒')
        self._task_resumer_manager.save()
        return output_dir

    def _process_single_video(self, each_resumer: TaskResumer, finished_video_path_list: list[Path],
                              video_info: VideoInfo) -> Path:
        self._processor_global_var.get_data()['fps'] = video_info.fps
        self._processor_global_var.get_data()['total_frames'] = video_info.frame_count
        self._processor_global_var.get_data()['width'] = video_info.width
        self._processor_global_var.get_data()['height'] = video_info.height
        if video_info.crop: # 自定义函数
            self._update_processor_global_var_with_crop_info(video_info.crop.x,
                                                             video_info.crop.y,
                                                             video_info.crop.w,
                                                             video_info.crop.h)
        else:
            self._update_processor_global_var_with_crop_info()
        engine_type: VideoProcessEngine = cfg.get(cfg.video_process_engine)
        finished_video_path: Path = self._video_handler.process_video(video_info.video_path, engine_type) # # 处理视频*
        finished_video_path_list.append(finished_video_path)
        each_resumer.output_video_path = finished_video_path  # 每一个已经处理完成的视频的路径,用来任务判断是否完成
        self._signal_bus.advance_total_progress.emit(1)
        self._task_resumer_manager.save() # 更新任务状态
        loguru.logger.info(f'处理视频{video_info.video_path}完成')
        return finished_video_path

    def _get_black_remove_algorithm(self) -> BlackRemoveAlgorithm | None:
        # 根据配置获取黑边处理算法，返回算法
        black_remove_algorithm_enum: BlackBorderAlgorithm = cfg.get(cfg.video_black_border_algorithm)
        match black_remove_algorithm_enum:
            case BlackBorderAlgorithm.Dynamic:
                black_remove_algorithm_impl = VideoRemover()
            case BlackBorderAlgorithm.Static:
                black_remove_algorithm_impl = IMGBlackRemover()
            case BlackBorderAlgorithm.Disable:
                black_remove_algorithm_impl = None
            case _:
                raise ValueError(f"不支持的黑边去除算法{black_remove_algorithm_enum}")
        return black_remove_algorithm_impl

    def _update_processor_global_var_with_crop_info(self, x: int | None = None,
                                                    y: int | None = None,
                                                    width: int | None = None,
                                                    height: int | None = None):
        self._processor_global_var.get_data()['crop_x'] = x
        self._processor_global_var.get_data()['crop_y'] = y
        self._processor_global_var.get_data()['crop_width'] = width
        self._processor_global_var.get_data()['crop_height'] = height

    def _get_video_resolution(self,
                              video_info_list: list[VideoInfo],
                              video_orientation: Orientation) -> tuple[int, int]:
    # 为所有输入视频确定一个统一的目标输出分辨率​​，而不是为每个视频单独计算不同的分辨率。
        def get_best_resolution(video_info_list: list[VideoInfo],
                                video_orientation: Orientation) -> tuple[int, int]:
            def get_most_compatible_resolution(video_info_list: list[VideoInfo],
                                               orientation: Orientation) -> tuple[int, int]:
                """获取最合适的视频分辨率"""
                resolutions: list[tuple[int, int]] = []
                # 获取每个视频的有效分辨率
                for each in video_info_list:
                    # 优先使用裁剪后的分辨率（crop.w和 crop.h）,如果视频未被裁剪，则使用原始分辨率（width和 height）
                    width, height = (each.crop.w, each.crop.h) if each.crop else (each.width, each.height)

                    # 判断视频的方向,如果视频的方向和用户选择的方向不一致则需要调换宽高
                    if (orientation == Orientation.HORIZONTAL and width > height) or (
                            orientation == Orientation.VERTICAL and width < height):
                        resolutions.append((width, height))
                    else:
                        resolutions.append((height, width))

                aspect_ratios: list[float] = [i[0] / i[1] for i in resolutions] # 计算所有视频的宽高比
                most_common_ratio = Counter(aspect_ratios).most_common(1)[0][0] # 找出最常见的宽高比
                compatible_resolutions = [res for res in resolutions if (res[0] / res[1]) == most_common_ratio] # 筛选具有该宽高比的所有分辨率
                compatible_resolutions.sort(key=lambda x: (x[0] * x[1]), reverse=True) # 按像素总量（宽×高）降序排序
                width, height = compatible_resolutions[0][:2] # 选择像素量最大的分辨率作为最佳分辨率
                return width, height

            loguru.logger.debug('正在获取最佳分辨率')
            self._signal_bus.set_total_progress_description.emit("调优参数")
            self._signal_bus.set_total_progress_reset.emit()
            self._signal_bus.set_detail_progress_reset.emit()
            best_width, best_height = get_most_compatible_resolution(video_info_list, video_orientation)
            loguru.logger.info(f'最佳分辨率获取完成,最佳分辨率为: {best_width}x{best_height}')
            return best_width, best_height

        resolution_mode: VideoResolution = cfg.get(cfg.video_resolution) # 从配置中读取用户设置的分辨率模式（如自动/480P/720P等）
        match resolution_mode:
            case VideoResolution.Auto:
                return get_best_resolution(video_info_list, video_orientation)
            case VideoResolution.P480 if video_orientation == Orientation.HORIZONTAL:
                return 720, 480
            case VideoResolution.P480 if video_orientation == Orientation.VERTICAL:
                return 480, 720
            case VideoResolution.P720 if video_orientation == Orientation.HORIZONTAL:
                return 1280, 720
            case VideoResolution.P720 if video_orientation == Orientation.VERTICAL:
                return 720, 1280
            case VideoResolution.P1080 if video_orientation == Orientation.HORIZONTAL:
                return 1920, 1080
            case VideoResolution.P1080 if video_orientation == Orientation.VERTICAL:
                return 1080, 1920
            case VideoResolution.P1440 if video_orientation == Orientation.HORIZONTAL:
                return 2560, 1440
            case VideoResolution.P1440 if video_orientation == Orientation.VERTICAL:
                return 1440, 2560
            case VideoResolution.P2160 if video_orientation == Orientation.HORIZONTAL:
                return 3840, 2160
            case VideoResolution.P2160 if video_orientation == Orientation.VERTICAL:
                return 2160, 3840
            case VideoResolution.P4320 if video_orientation == Orientation.HORIZONTAL:
                return 7680, 4320
            case VideoResolution.P4320 if video_orientation == Orientation.VERTICAL:
                return 4320, 7680
            case _:
                raise ValueError(f"不支持的分辨率{resolution_mode}")


if __name__ == '__main__':
    p = ProgramCoordinator()
    print(p.process([
        Path(r"E:\load\python\Project\VideoFusion\TempAndTest\other\video_2024-03-04_16-55-20.mp4"),
        Path(r"E:\load\python\Project\VideoFusion\测试\dy\b7bb97e21600b07f66c21e7932cb7550.mp4")
    ],
))
