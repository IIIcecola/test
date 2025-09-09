from pathlib import Path

import cv2

from src.common.black_remove_algorithm.black_remove_algorithm import BlackRemoveAlgorithm
from src.common.black_remove_algorithm.img_black_remover import IMGBlackRemover
from src.core.datacls import CropInfo, VideoInfo # **


class VideoInfoReader:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
    '''
    输入：
      black_remove_algorithm: 可选的黑边去除算法实例
      crop_enabled: 是否启用裁剪（默认True）
    输出：
      VideoInfo对象（包含视频元数据和裁剪信息）
    '''
    def get_video_info(self,
                       black_remove_algorithm: BlackRemoveAlgorithm | None,
                       crop_enabled: bool = True) -> VideoInfo:
        # 获取基础视频属性​
        video = cv2.VideoCapture(str(self.video_path))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()
                         
        # 如果禁用裁剪或未提供算法，直接返回基础视频信息（无裁剪数据）
        if not crop_enabled:
            return VideoInfo(video_path=self.video_path,
                             fps=fps,
                             frame_count=frame_count,
                             width=width,
                             height=height)

        if black_remove_algorithm is None:
            return VideoInfo(video_path=self.video_path,
                             fps=fps,
                             frame_count=frame_count,
                             width=width,
                             height=height)

        # 获取剪裁信息；调用算法的remove_black()方法
        x, y, w, h = black_remove_algorithm.remove_black(self.video_path)
        if w == width and h == height: # 裁剪尺寸与原视频相同 → 无实际裁剪
            return VideoInfo(video_path=self.video_path,
                             fps=fps,
                             frame_count=frame_count,
                             width=width,
                             height=height)
        elif x == 0 and y == 0 and w == 0 and h == 0: # 算法返回零区域 → 视为无效结果
            return VideoInfo(video_path=self.video_path,
                             fps=fps,
                             frame_count=frame_count,
                             width=width,
                             height=height)
        return VideoInfo(video_path=self.video_path, # 有效裁剪​​：将CropInfo存入VideoInfo
                         fps=fps,
                         frame_count=frame_count,
                         width=width,
                         height=height,
                         crop=CropInfo(x, y, w, h))

    def get_crop_info(self, black_remove_algorithm: BlackRemoveAlgorithm) -> CropInfo:
        # 专用接口​​：直接获取裁剪信息（不包含视频元数据）
        x, y, w, h = black_remove_algorithm.remove_black(self.video_path)
        return CropInfo(x, y, w, h)


if __name__ == '__main__':
    v = VideoInfoReader(r"E:\load\python\Project\VideoFusion\tests\test_data\videos\001.mp4")
    print(v.get_video_info(IMGBlackRemover()))
