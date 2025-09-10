from pathlib import Path

import cv2
# FFmpeg操作封装
from src.common.ffmpeg_handler import FFmpegHandler
# 各种处理器管理器（音频、外部程序、OpenCV）
from src.common.processors.audio_processors.audio_processor_manager import AudioProcessorManager
from src.common.processors.exe_processors.exe_processor_manager import EXEProcessorManager
from src.common.processors.opencv_processors.opencv_processor_manager import OpenCVProcessorManager
# 全局变量存储
from src.common.processors.processor_global_var import ProcessorGlobalVar
# 视频处理引擎（FFmpeg和OpenCV实现）
from src.common.video_engines.ffmpeg_video_engine import FFmpegVideoEngine
from src.common.video_engines.opencv_video_engine import OpenCVVideoEngine
from src.config import VideoProcessEngine
from src.core.enums import Orientation
# 信号总线（用于进度更新）
from src.signal_bus import SignalBus
# 工具函数（临时目录和输出路径生成）
from src.utils import TempDir, get_output_file_path


class VideoHandler:
    # 这个类负责协调视频处理流程，支持多种处理引擎和处理器。
    def __init__(self):
        self._signal_bus: SignalBus = SignalBus()
        self._temp_dir: TempDir = TempDir() # 临时目录管理
        self._ffmpeg_handler: FFmpegHandler = FFmpegHandler() # FFmpeg操作封装
        self._open_cv_video_engine: OpenCVVideoEngine = OpenCVVideoEngine()
        self._ffmpeg_video_engine: FFmpegVideoEngine = FFmpegVideoEngine()
        self._audio_processor_manager: AudioProcessorManager = AudioProcessorManager()
        self._video_processor_manager: OpenCVProcessorManager = OpenCVProcessorManager()
        self._exe_processor_manager: EXEProcessorManager = EXEProcessorManager()
        self._processor_global_var: ProcessorGlobalVar = ProcessorGlobalVar() # 全局变量存储
        self.is_running: bool = True

        self._signal_bus.set_running.connect(self._set_running)

    def process_video(self, input_video_path: Path,
                      engine_type: VideoProcessEngine = VideoProcessEngine.FFmpeg) -> Path:
        # 功能​​：处理单个视频文件
        '''
        参数​​:input_video_path: 输入视频路径;engine_type: 处理引擎类型（FFmpeg或OpenCV）
        ​​流程​​:1.根据引擎类型选择处理方式;2.使用外部程序处理器进行后处理;3.返回处理后的视频路径
        '''
        if engine_type == VideoProcessEngine.OpenCV:
            video_after_processed = self._open_cv_video_engine.process_video(input_video_path)
        elif engine_type == VideoProcessEngine.FFmpeg:
            video_after_processed = self._ffmpeg_video_engine.process_video(input_video_path)
        else:
            raise ValueError(f"不支持的视频处理引擎{engine_type}")

        video_after_processed = self._exe_processor_manager.process(video_after_processed)
        return video_after_processed

    def merge_videos(self, video_list: list[Path]) -> Path:
        return self._ffmpeg_handler.merge_videos(video_list) # 委托给FFmpegHandler合并多个视频文件

    def compress_video(self, input_video_path: Path) -> Path:
        return self._ffmpeg_handler.reencode_video(input_video_path) # 委托给FFmpegHandler视频压缩

    def _video_process(self, input_video_path: Path) -> Path:
        """
        读取输入视频，逐帧处理，然后写入输出视频,以及音频处理

        Args:
            input_video_path: 输入视频的路径
        """
        self._signal_bus.set_detail_progress_reset.emit()
        # 打开视频
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise ValueError("无法打开输入视频")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 因为处理后的视频的宽高可能会发生变化，所以需要重新获取宽高
        width, height = self._get_after_process_width_and_height(input_video_path)

        # MP4V比较通用，但是不支持透明度;创建输出视频
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        output_file_path = get_output_file_path(input_video_path, "video_processed")
        out = cv2.VideoWriter(str(output_file_path), fourcc, fps, (width, height))

        # 逐帧处理
        self._signal_bus.set_detail_progress_max.emit(total_frames)
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret or not self.is_running:
                break

            # 对帧进行处理
            processed_frame = self._video_processor_manager.process(frame)

            # 写入处理后的帧
            out.write(processed_frame)
            self._signal_bus.advance_detail_progress.emit(1)

        # 释放资源
        cap.release()
        out.release()

        self._signal_bus.set_detail_progress_finish.emit()
        return output_file_path

    def _audio_process(self, input_video_path: Path) -> Path:
        """
        读取输入视频，逐帧处理，然后写入输出视频,以及音频处理

        Args:
            input_video_path: 输入视频的路径
        """
        output_file_path = get_output_file_path(input_video_path, "audio_processed")
        self._audio_processor_manager.process(input_video_path)
        return output_file_path

    def _get_after_process_width_and_height(self, input_video_path: Path, ) -> tuple[int, int]:
        # 功能​​：获取处理后的视频尺寸
        # 方法​​：处理第一帧并获取其尺寸
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise ValueError("无法打开输入视频")

        ret, frame = cap.read()
        if not ret:
            raise ValueError("无法读取视频的第一帧")
        processed_frame = self._video_processor_manager.process(frame)

        height, width = processed_frame.shape[:2]
        cap.release()
        return width, height

    def _set_running(self, is_running: bool):
        self.is_running = is_running


if __name__ == '__main__':
    v = VideoHandler()
    global_var = ProcessorGlobalVar()
    global_var.update("crop_x", 0)
    global_var.update("crop_y", 515)
    global_var.update("crop_width", 716)
    global_var.update("crop_height", 482)
    global_var.update("rotation_angle", 90)
    global_var.update("orientation", Orientation.HORIZONTAL)
    global_var.update("target_width", 500)
    global_var.update("target_height", 300)

    print(v.process_video(
            Path(r"E:\load\python\Project\VideoFusion\TempAndTest\dy\b7bb97e21600b07f66c21e7932cb7550.mp4")))
