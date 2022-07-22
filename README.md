# mmwRanging
A Python package for linear distance measurement using ultra-wideband (UWB) millimeter-wave (mmWave) radar, based on
frequency modulated continuous wave (FMCW) technology. The package provides micrometer accuracy, and nanometer-level
jitter at medium range (>1m). It is particularly suitable for use with 2pi-labs' 2piSENSE radar systems.

## Citations
If the provided code helps your research, please cite the following articles on which the package is based:

1.  ```
    L. Piotrowsky, S. Kueppers, T. Jaeschke, N. Pohl,
    "Distance Measurement Using mmWave Radar: Micron Accuracy at Medium Range,"
    accepted for publication in IEEE Transactions on Microwave Theory and Techniques,
    Jul. 2022.
    ```
    [Accepted version](https://www.researchgate.net/publication/362175118_Distance_Measurement_Using_mmWave_Radar_Micron_Accuracy_at_Medium_Range)
2.  ```
    L. Piotrowsky, J. Barowski and N. Pohl,
    "Near-Field Effects on Micrometer Accurate Ranging With Ultra-Wideband mmWave Radar,"
    in IEEE Antennas and Wireless Propagation Letters, vol. 21, no. 5, pp. 938-942,
    May 2022, doi: 10.1109/LAWP.2022.3152558.
    ```
3.  ```
    L. Piotrowsky, T. Jaeschke, S. Kueppers, J. Siska and N. Pohl,
    "Enabling High Accuracy Distance Measurements With FMCW Radar Sensors,"
    in IEEE Transactions on Microwave Theory and Techniques, vol. 67, no. 12, pp. 5360-5371,
    Dec. 2019, doi: 10.1109/TMTT.2019.2930504.
    ```

## Installation
Please simply copy the package directory to your workspace, and install the requirements by running:
```
$ pip install -r requirements.txt
```

## Usage
[Here](./examples) you can find various examples.