# Squirrel Cage Motor Anomaly Detection

1. Thermovision
2. Vibration

## Dataset distibution

```
train = [
  "clutch_2/current-load-0A", "clutch_2/current-load-2A", 
  "clutch_2/current-load-4A", "clutch_2/current-load-6A", 
  "clutch_2/misalignment-current-load-0A", "clutch_2/misalignment-current-load-2A",
  "clutch_2/misalignment-current-load-4A", "clutch_2/misalignment-current-load-6A", 
  "clutch_2/misalignment-3-current-load-0A", "clutch_2/misalignment-3-current-load-2A", 
  "clutch_2/misalignment-3-current-load-4A", "clutch_2/misalignment-3-current-load-6A", 
  "clutch_2/rotor-1-current-load-0A-clutch-tightened", "clutch_2/rotor-1-current-load-2A", 
  "clutch_2/rotor-1-current-load-6A", "clutch_2/rotor-3-current-load-0A", 
  "clutch_2/rotor-3-current-load-2A", "clutch_2/rotor-3-current-load-6A",
  "clutch_2/rotor-6-current-load-0A", "clutch_2/rotor-6-current-load-2A", 
  "clutch_2/rotor-6-current-load-6A", "clutch_1/start-up-current-load-0A", 
  "clutch_1/current-load-6A", "clutch_1/misalignment-current-load-0A", 
  "clutch_1/misalignment-current-load-6A",
]

valid = [
  "clutch_2/current-load-0A-2", "clutch_2/current-load-4A-2",
  "clutch_2/misalignment-2-current-load-0A", "clutch_2/misalignment-2-current-load-4A",
  "clutch_2/rotor-1-current-load-0A", "clutch_2/rotor-3-current-load-4A",
  "clutch_1/current-load-2A", "clutch_1/misalignment-current-load-2A",
]
    
test = [
  "clutch_2/current-load-2A-2", "clutch_2/current-load-6A-2",
  "clutch_2/misalignment-2-current-load-2A", "clutch_2/misalignment-2-current-load-6A",
  "clutch_2/rotor-1-current-load-4A", "clutch_2/rotor-6-current-load-4A",
  "clutch_1/start-up-current-load-0A-2", "clutch_1/current-load-4A",
  "clutch_1/misalignment-current-load-4A",
]
```

## Usage

* keys
  ```commandline
    export NEPTUNE_API_TOKEN=""
    export NEPTUNE_PROJECT_NAME=""
  ```
  or call `nepptune_exports.sh` from main directory.
  
* config - available at [configs directory](./configs).

* data - available at [chmura.put.poznan.pl](https://chmura.put.poznan.pl/s/t1VhZlh9sOdyl4Z).
  
* run train
  ```commandline
  HYDRA_FULL_ERROR=1 python run.py --config-name thermo_config name=exp
  ```
  
* run eval
  ```commandline
  HYDRA_FULL_ERROR=1 python run.py --config-name thermo_config name=exp eval_mode=True trainer.resume_from_checkpoint=/home/path/to/model
  ```

* run export to ONNX
  ```commandline
  HYDRA_FULL_ERROR=1 python run.py --config-name thermo_config name=exp eval_mode=True trainer.resume_from_checkpoint=/home/path/to/model export.export_to_onnx=True
  ```

