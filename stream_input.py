import subprocess
import sys
import numpy as np

def decode_mp4_with_ffmpeg(input_path, width, height):
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-vf', f'scale={width}:{height}',
        'pipe:1'
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = width * height * 3

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break 

        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
        yield frame

    proc.stdout.close()
    stderr = proc.stderr.read().decode('utf-8', errors='ignore')
    retcode = proc.wait()
    proc.stderr.close()
    if retcode != 0:
        print("FFmpeg завершился с ошибкой:", retcode, file=sys.stderr)
        print(stderr, file=sys.stderr)
        raise RuntimeError("FFmpeg error")