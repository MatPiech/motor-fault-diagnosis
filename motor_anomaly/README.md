# Squirrel Cage Motor Anomaly Detection

## Methods

1. [Thermovision](./thermovision/)
2. [Vibrations](./vibrations)

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
