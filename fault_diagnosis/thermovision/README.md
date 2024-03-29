# Fault Diagnosis in a Squirrel-Cage Induction Motor using Thermal Images

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
 
* data - images are available at at [zenodo](https://doi.org/10.5281/zenodo.8203070) and [chmura.put.poznan.pl](https://chmura.put.poznan.pl/s/zwn7VaVgV3FI2ER) in *workswell_wic_640* and *flir_lepton_3_5* directories.

* config - available at [configs directory](./configs).

* [neptune.ai](https://neptune.ai/) logging keys should be exported as below, otherwise disable logger in config by setting `logger: False`
  ```shell
    export NEPTUNE_API_TOKEN=""
    export NEPTUNE_PROJECT_NAME=""
  ```
  
* train
  ```shell
  HYDRA_FULL_ERROR=1 python run.py --config-name thermo_config name=exp
  ```
  
* evaluation
  ```shell
  HYDRA_FULL_ERROR=1 python run.py --config-name thermo_config name=exp eval_mode=True trainer.resume_from_checkpoint=/home/path/to/model
  ```

* model export to ONNX format
  ```shell
  HYDRA_FULL_ERROR=1 python run.py --config-name thermo_config name=exp eval_mode=True trainer.resume_from_checkpoint=/home/path/to/model export.export_to_onnx=True
  ```

## Results

<p align="center">
  <img width="900" src="../../.images/acc_infer_size_comparison.png">
</p>

<p align="center">
  <img width="900" src="../../.images/captum_interpretability_models.png">
</p>
