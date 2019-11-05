LIMITS: LIghtweight Machine learning for IoT Systems
========

<span style="color:red">Preliminary Beta Version</span>

**LIMITS** is python-based open source framework for automating *high-level* machine learning tasks targeted at resource-constrained IoT platforms. The *low-level* trainining of the models is performed by the coupled **WEKA** framework. *LIMITS* parses the *WEKA* outputs and derives an abstract model representation which is utilized for *C/C++* code generation. Moreover, *LIMITS* can explicitly integrate the compilation toolchain of the targeted IoT platform in order to derive accurate assessments of the required memory resources for deploying the model to the considered platform.

### Machine Learning Models
Currently, the following models can be utilized for data analysis and code generation [Classification/Regression Support]:
- *Artificial Neural Networks (ANN)* [C/R] with sigmoid activation function
- *M5 Regression Tree* [R]
- *Random Forest (RF)*  [C/R]
- *Linear Support-Vector Machine (SVM)* [C/R] based on Sequential Minimal Optimization (SMO)

Additional models are planned for later releases.

### Assumptions
- *LIMITS* works with CSV input data with a single line header. The label attribute of the data set should be defined in the **first column**, all other columns represent feature attributes. Examples can be found in [examples](src/), example data sets are provided at [data](examples/).
- If the label is represented by a *string* value, *LIMITS* will perform **classification**, otherwise **regression**.


### Quickstart
After following the [setup instructions](INSTALL.md), the Command Line Interface (CLI) can be utilized for a fast setup verification:

```
$ ./cli.py -r ../examples/mnoA.csv -m rf,m5,ann

r2               mae              rmse             training         test                                                       
0.788+/-0.014    2.881+/-0.173    3.935+/-0.153    0.204+/-0.016    0.001+/-0.003
0.772+/-0.030    2.773+/-0.081    4.022+/-0.206    0.546+/-0.007    0.004+/-0.008
0.791+/-0.022    2.978+/-0.250    4.060+/-0.254    3.150+/-0.011    0.003+/-0.006
```

