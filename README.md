# PatchVQ-docker
Docker inference support for using Patch-VQ (‘Patching Up’ the Video Quality Problem) to compute mean opinion score (MOS) on your
own video files. A build of this container has been pushed to Docker Hub already (https://hub.docker.com/r/pixop/patchvq/) for your convenience.

Forked from: https://github.com/baidut/PatchVQ 

***Please email yingzhenqiang at gmail dot com for any questions about PatchVQ***

## Run CUDA accelerated inference on your own video file

Computing MOS on `video.mp4` located in the current directory:

```docker run --gpus all -it --ipc=host --rm -v $(pwd):/mnt/host pixop/patchvq:latest /mnt/host/video.mp4```

Note: Requires Nvidia Container Toolkit to be installed (https://github.com/NVIDIA/nvidia-docker).

## Run CPU inference on your own video file

Computing MOS on `video.mp4` located in the current directory:

```docker run -it --ipc=host --rm -v $(pwd):/mnt/host pixop/patchvq:latest /mnt/host/video.mp4```

## Run inference a cropped, 8 seconds clip of your own video file

Computing MOS on `video.mp4` located in the current directory while using FFmpeg pre-processing to crop and stop processing at 8 seconds:

```docker run -it --ipc=host --rm -v $(pwd):/mnt/host pixop/patchvq:latest /mnt/host/video.mp4 "-vf crop=1280:720 -t 8"```

Note: It is possible to supply any number of arguments to FFmpeg this way.

## Notes

1. The implementation is not production quality by any means and is merely designed to be a least effort way to compute the MOS. No error checking of any kind is performed!
2. The original implementation was tweaked a bit to output PNGs instead of JPEGs to prevent the MOS drop due to lossy encoding. Saving JPEGs at the highest quality still produces noticeable degradation.
3. Tested on 64-bit Linux.
