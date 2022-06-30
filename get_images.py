from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np


def raw_to_8bit(data: np.ndarray, rgb_color: bool = False) -> np.ndarray:
    data_norm = np.zeros_like(data)
    cv2.normalize(data, data_norm, 0, 65535, cv2.NORM_MINMAX)
    data_8bit = np.right_shift(data_norm, 8)

    if rgb_color:
        return cv2.cvtColor(np.uint8(data_8bit), cv2.COLOR_GRAY2RGB)
    else:
        return np.uint8(data_8bit)


def convert_to_celsius_degrees(data: np.ndarray) -> np.ndarray:
    temperature_data = data / 100 - 273.15

    return temperature_data


@click.command()
@click.option('--video_source', type=int, default=2, help='Input video / camera source')
@click.option('--save', is_flag=True, help='Saves registered images')
@click.option('--show', is_flag=True, help='Shows registered images')
def get_images(video_source, save, show):
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    if save:
        now = datetime.now()
        dir_name = Path('data', now.strftime("%Y_%m_%d_%H_%M_%S"))
        dir_name.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('./data/'+now.strftime("%Y_%m_%d_%H_%M_%S")+'.mp4', fourcc, 8.6, (160, 120), isColor=False)
    
    while True:
        ret, frame = cap.read()

        frame_8bit = raw_to_8bit(frame)
        # frame_temperature = convert_to_celsius_degrees(frame)

        if save:
            img_path = dir_name / f'{datetime.now().strftime("%H_%M_%S_%f")}.png'
            cv2.imwrite(str(img_path), frame)
            out.write(frame_8bit)

        if show:
            cv2.imshow('uint8', frame_8bit)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    get_images()
