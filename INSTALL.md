Setup Instructions
========

## WEKA Setup (required)
- Download and install the [WEKA framework](https://www.cs.waikato.ac.nz/ml/weka/downloading.html)
- Define *self.wekaPath* in *src/settings/Settings.py* for *[path]/WEKA/weka.jar*

## Python (required)
- Download and install [Python 3.X](https://www.python.org/downloads/)
- Install [pip](https://pip.pypa.io/en/stable/installing/)
- Install the required packages
```
$ pip install matplotlib
$ pip install numpy
```

## Target IoT Platforms (optional)

### MSP430
- Download and install the [Code Composer Studio (CCS) IDE](http://www.ti.com/tool/CCSTUDIO)
- Within *CCS*, build a dummy project and configure the target MSP430 model
- Copy the generated *lnk_msp320_[model].cmd* file to the *src* folder of *LIMITS*
- Define *self.mspPath* in *src/settings/Settings.py* for *[path]/ccs910/ccs*
- Download  [tiny printf](http://www.msp430launchpad.com/2012/06/using-printf.html) and copy it into the *src/*
 folder

### Atmega328
- Download and install the [Arduino IDE](https://www.arduino.cc/en/main/software)
- **Important:** *LIMITS* can only parse the outputs of the English Arduino IDE
- Define *self.arduinoPath* in *src/settings/Settings.py* for *[path]/Arduino/arduino.exe*

### ESP32
- Download and install the [Espressif IoT Development Framework](https://github.com/espressif/esp-idf)
- Set up the *hello_world* project according to the [Gettting Started Manual](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/) and test the compilation process. *LIMITS* will use the project build and deploy its own ESP32 code
- Define *self.espPath* in *src/settings/Settings.py* for *[path]/esp-idf/tools/idf.py*
- Define *self.espProject* in *src/settings/Settings.py* for *[path]/esp/hello_world*
