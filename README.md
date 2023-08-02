# Fault Diagnosis in a Squirrel-Cage Induction Motor

> This is the official repository of the paper: [*Unraveling Induction Motor State through Thermal Imaging and Edge Processing: A Step towards Explainable Fault Diagnosis*](https://ein.org.pl/Unraveling-Induction-Motor-State-through-Thermal-Imaging-and-Edge-Processing-A-Step,170114,0,2.html).


<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br/>
The Squirrel Cage Induction Motor Fault Diagnosis Dataset is a multi-sensor data collection gathered to expand research on anomaly detection, fault diagnosis, and predictive maintenance, mainly using non-invasive methods such as thermal observation or vibration measurement. The measurements were gathered using advanced laboratory at Wrocław University of Science and Technology, designed to simulate and study motor defects. The collected dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a> whereas accompanying scripts and source code are licensed under a [MIT License](./LICENSE).

## Citation

```
@Article{Piechocki2023,
  journal="Eksploatacja i Niezawodność – Maintenance and Reliability",
  issn="1507-2711",
  volume="25",
  number="3",
  year="2023",
  title="Unraveling Induction Motor State through Thermal Imaging and Edge Processing: A Step towards Explainable Fault Diagnosis",
  abstract="Equipment condition monitoring is essential to maintain the reliability of the electromechanical systems. Recently topics related to fault diagnosis have attracted significant interest, rapidly evolving this research area. This study presents a non-invasive method for online state classification of a squirrel-cage induction motor. The solution utilizes thermal imaging for non-contact analysis of thermal changes in machinery. Moreover, used convolutional neural networks (CNNs) streamline extracting relevant features from data and malfunction distinction without defining strict rules. A wide range of neural networks was evaluated to explore the possibilities of the proposed approach and their outputs were verified using model interpretability methods. Besides, the top-performing architectures were optimized and deployed on resource-constrained hardware to examine the system's performance in operating conditions. Overall, the completed tests have confirmed that the proposed approach is feasible, provides accurate results, and successfully operates even when deployed on edge devices.",
  author="Piechocki, Mateusz and Pajchrowski, Tomasz and Kraft, Marek and Wolkiewicz, Marcin and Ewert, Paweł",
  doi="10.17531/ein/170114",
  url="https://doi.org/10.17531/ein/170114"
}
```

## Sensors

1. [Workswell InfraRed Camera (WIC) 640](https://workswell-thermal-camera.com/workswell-infrared-camera-wic/) with 640x512 px resolution.
2. [Flir Lepton 3.5](https://www.flir.com/products/lepton/?model=500-0771-01&vertical=microcam&segment=oem) (160x120 px resolution IR sensor) with [PureThermal 2 Smart I/O Module](https://cdn.sparkfun.com/assets/c/4/7/8/4/PureThermal_2_-_Datasheet_-_1.2.pdf)
3. [Triaxial DeltaTron Accelerometer Type 4506](https://www.bksv.com/en/transducers/vibration/accelerometers/ccld-iepe/4506-b-003)
4. [Arduino Nano 33 BLE Sense](https://docs.arduino.cc/hardware/nano-33-ble-sense) with [LSM9DS1](https://content.arduino.cc/assets/Nano_BLE_Sense_lsm9ds1.pdf) IMU and [MP34DT05](https://content.arduino.cc/assets/Nano_BLE_Sense_mp34dt05-a.pdf) Omnidirectional Digital Microphone.


## Dataset

<a href="https://doi.org/10.5281/zenodo.8203070"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8203070.svg" alt="DOI"></a>

The collected data is publicly available under _Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License_ at [zenodo](https://doi.org/10.5281/zenodo.8203070) and [chmura.put.poznan.pl](https://chmura.put.poznan.pl/s/zwn7VaVgV3FI2ER).

### Data and structure
The Squirrel-Cage Induction Motor Fault Diagnosis dataset consists of several, simultaneously collected signals, such as:

<details close>
<summary>640x512 px thermal images in <i>workswell_wic_640</i> directory (Workswell WIC 640)</summary>
<p align="center">
  <img width="900" height="300" src="./.images/workswell_wic_640_thermal_images.png">
</p>
</details>

<details close>
<summary>160x120 px thermal images in <i>flir_lepton_3_5</i> folder (Flir Lepton 3.5)</summary>
<p align="center">
  <img width="600" height="300" src="./.images/flir_lepton_3_5_thermal_images.png">
</p>
</details>

<details close>
<summary>Current and voltage signals with XXX Hz sampling rate in <i>sig_*_R_U_W.tdms</i> files</summary>

</details>

<details close>
<summary>Vibration data with 1000 Hz sampling rate in <i>vib_*_R_U_W.tdms</i> files (Triaxial DeltaTron)</summary>
<p align="center">
  <img width="800" height="600" src="./.images/Triaxial_DeltaTron_acc_data.png">
</p>
</details>

<details close>
<summary>IMU data with 100 Hz sampling rate in <i>imu_*.cbor</i> files (LSM9DS1)</summary>
<p align="center">
  <img width="800" height="600" src="./.images/LSM9DS1_acc_data.png">
</p>
</details>

<details close>
<summary>Microphone sound with 16 kHz sampling rate in <i>micro_*.json</i> files (MP34DT05)</summary>
<p align="center">
  <img width="800" height="600" src="./.images/MP34DT05_micro_data.png">
</p>
</details>

The dataset separates the use of different clutches and within them, experiments are split into 3 classes, according to the below description:
- `misalignment-X-*` - where `X` means a series of experiments with the same shift
- `rotor-X-*` - where `X` means the number of broken cages in the squirrel-cage rotor
- the other contains samples gathered during proper motor operation

All examinations were conducted with and without current load - in the range 0 - 6 A (`*-current-load-X`).


## Extract data in Python

<details close>
<summary>Thermal images</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


img_raw = np.asarray(Image.open(filepath), dtype=np.uint16)
img = normalize(img_raw)
plt.imshow(img, cmap='gray')
```
</details>

<details close>
<summary>LabVIEW tdms data</summary>

```python
import matplotlib.pyplot as plt
import pandas as pd
from nptdms import TdmsFile


tdms_file = TdmsFile.read(filepath)
df = tdms_file.as_dataframe()

df.plot()
```
</details>

<details close>
<summary>IMU cbor files</summary>

```python
import cbor2
import matplotlib.pyplot as plt
import numpy as np


with open(filepath, 'rb') as f:
    data = cbor2.decoder.load(f)

data = np.array(data['payload']['values'])
print(data.shape)

plt.plot(data)
```
</details>

<details close>
<summary>Microphone JSON files</summary>

```python
import json

import matplotlib.pyplot as plt
import numpy as np


with open(filepath, 'r') as f:
    data = json.load(f)

plt.plot(data['payload']['values'])
```
</details>
