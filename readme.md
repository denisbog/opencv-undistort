# run

```bash
cargo r --release -- calibrate --calibration-dir calibration --calibration-file calib.bin
cargo r --release -- correct --calibration-file calib.bin --correction-dir process --output-dir out
```
