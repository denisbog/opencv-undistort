# run

```bash
cargo r --release -- calibrate --calibration-dir calibration --calibration-file calib.bin
cargo r --release -- correct --calibration-file calib.bin --correction-dir process --output-dir out
cargo r --release -- solve --calibration-file calib.bin --calibration-file calib.bin --image-dir out
```

## video for linux

```bash
v4l2-ctl --device /dev/video4 --set-fmt-video=pixelformat=MJPG
v4l2-ctl --all -d /dev/video4 --list-formats
```
