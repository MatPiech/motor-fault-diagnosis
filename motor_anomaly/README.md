# Squirrel Cage Motor Anomaly Detection

1. Thermovision
2. Vibration

## Dataset distibution

```
train = [
    'current-load-0A', 'current-load-2A', 'current-load-4A', 'current-load-6A',
    'misalignment-current-load-0A', 'misalignment-current-load-2A', 'misalignment-current-load-4A', 'misalignment-current-load-6A',
    'misalignment-3-current-load-0A', 'misalignment-3-current-load-2A', 'misalignment-3-current-load-4A', 'misalignment-3-current-load-6A',
    'rotor-1-current-load-0A-clutch-tightened', 'rotor-1-current-load-2A', 'rotor-1-current-load-6A',
    'rotor-3-current-load-0A', 'rotor-3-current-load-2A', 'rotor-3-current-load-6A',
    'rotor-6-current-load-0A', 'rotor-6-current-load-2A', 'rotor-6-current-load-6A'
]

valid = [
    'current-load-0A-2', 'current-load-4A-2',
    'misalignment-2-current-load-0A', 'misalignment-2-current-load-4A',
    'rotor-1-current-load-0A', 'rotor-3-current-load-4A'
]    

test = [
    'current-load-2A-2', 'current-load-6A-2',
    'misalignment-2-current-load-2A', 'misalignment-2-current-load-6A'
    'rotor-1-current-load-4A', 'rotor-6-current-load-4A'
]
```
  