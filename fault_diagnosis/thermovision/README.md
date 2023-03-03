# Squirrel Cage Motor Fault Diagnosis in Thermal Images

## Sensors & Data

1. [Workswell InfraRed Camera (WIC) 640](https://workswell-thermal-camera.com/workswell-infrared-camera-wic/) with 640x512 px resolution.

<p align="center">
  <img width="900" height="300" src="../../.images/workswell_wic_640_thermal_images.png">
</p>

2. [Flir Lepton 3.5](https://www.flir.com/products/lepton/?model=500-0771-01&vertical=microcam&segment=oem) (160x120 px resolution IR sensor) with [PureThermal 2 Smart I/O Module](https://cdn.sparkfun.com/assets/c/4/7/8/4/PureThermal_2_-_Datasheet_-_1.2.pdf)

<p align="center">
  <img width="600" height="300" src="../../.images/flir_lepton_3_5_thermal_images.png">
</p>


## Usage
 
* data - images are available at [chmura.put.poznan.pl](https://chmura.put.poznan.pl/s/t1VhZlh9sOdyl4Z) in *workswell_wic_640* and *flir_lepton_3_5* directories.

* config - available at [configs directory](./configs).

* [neptune.ai](https://neptune.ai/) logging keys:
  ```commandline
    export NEPTUNE_API_TOKEN=""
    export NEPTUNE_PROJECT_NAME=""
  ```
  
* train
  ```commandline
  HYDRA_FULL_ERROR=1 python run.py --config-name thermo_config name=exp
  ```
  
* evaluation
  ```commandline
  HYDRA_FULL_ERROR=1 python run.py --config-name thermo_config name=exp eval_mode=True trainer.resume_from_checkpoint=/home/path/to/model
  ```

* model export to ONNX format
  ```commandline
  HYDRA_FULL_ERROR=1 python run.py --config-name thermo_config name=exp eval_mode=True trainer.resume_from_checkpoint=/home/path/to/model export.export_to_onnx=True
  ```