from pathlib import Path

from src.common.ffmpeg_handler import FFmpegHandler
from src.common.processors.audio_processors.audio_ffmpeg_processor import AudioFFmpegProcessor
from src.common.processors.base_processor import AudioProcessor, AudioProcessorManager as APM


class AudioProcessorManager(APM):
    # 协调多个音频处理器的执行顺序，形成一个处理流水线。
    def __init__(self):
        super().__init__()
        self._ffmpeg_handler = FFmpegHandler() # 创建 FFmpeg 处理工具实例
        self._audio_ffmpeg_processor = AudioFFmpegProcessor()

        self._processors: list[AudioProcessor] = [ # 返回类型：list[AudioProcessor]（音频处理器列表）
                self._audio_ffmpeg_processor
                ]

    def get_processors(self) -> list[AudioProcessor]:
        return self._processors

    def add_processor(self, processor: AudioProcessor): # processor: AudioProcessor：必须是 AudioProcessor类型（或其子类）的实例
        self._processors.append(processor)

    def process(self, x: Path) -> Path: #  输入音频文件的路径（Path对象）
        for processor in self._processors:
            x = processor.process(x)
        return x
