LIMITS: LIghtweight Machine learning for IoT Systems
========
**Preliminary Beta Version**

**LIMITS** is python-based open source framework for automating *high-level* machine learning tasks targeted at resource-constrained IoT platforms. The *low-level* trainining of the models is performed by the coupled **WEKA** framework. *LIMITS* parses the *WEKA* outputs and derives an abstract model representation which is utilized for *C/C++* code generation. Moreover, *LIMITS* can explicitly integrate the compilation toolchain of the targeted IoT platform in order to derive accurate assessments of the required memory resources for deploying the model to the considered platform.

- [**SETUP INSTRUCTIONS**](INSTALL.md)
- [**EXAMPLE APPLICATIONS**](EXAMPLES.md)


## Machine Learning Models
Currently, the following models can be utilized for data analysis and code generation [Classification/Regression Support]:
- *Artificial Neural Network (ANN)* [C/R] with sigmoid activation function
- *M5 Regression Tree* [R]
- *Random Forest (RF)*  [C/R]
- *Linear Support-Vector Machine (SVM)* [C/R] based on Sequential Minimal Optimization (SMO)

The integration of additional models is planned for later releases.

## Assumptions
- *LIMITS* works with CSV input data with a single line header. The label attribute of the data set should be defined in the **first column**, all other columns represent feature attributes. Examples can be found in [examples](src/), example data sets are provided at [data](examples/).
- If the label is represented by a *string* value, *LIMITS* will perform **classification**, otherwise **regression**.


## Quickstart
After following the [setup instructions](INSTALL.md), the Command Line Interface (CLI) can be utilized for a fast setup verification:

```
$ ./cli.py -r ../examples/mnoA.csv -m ann,m5,rf,svm

r2               mae              rmse             training         test                                                       
0.790+/-0.030    2.806+/-0.149    3.955+/-0.228    12.151+/-0.183   0.000+/-0.00
0.772+/-0.030    2.773+/-0.081    4.022+/-0.206    0.584+/-0.005    0.000+/-0.00
0.834+/-0.014    2.428+/-0.092    3.435+/-0.130    2.218+/-0.029    0.056+/-0.00
0.552+/-0.030    4.351+/-0.147    5.666+/-0.192    10.667+/-2.033   0.000+/-0.00
```


## Related Publications
- B. Sliwa, C. Wietfeld, [**Towards Data-driven Simulation of End-to-end Network Performance Indicators**](https://arxiv.org/abs/1904.10179), In *2019 IEEE 90th Vehicular Technology Conference (VTC-Fall)*, 2019
- B. Sliwa, C. Wietfeld, [**Empirical Analysis of Client-based Network Quality Prediction in Vehicular Multi-MNO Networks**](https://arxiv.org/abs/1904.10177), In *2019 IEEE 90th Vehicular Technology Conference (VTC-Fall)*, 2019
- B. Sliwa, R. Falkenberg, T. Liebig, N. Piatkowski, C. Wietfeld, [**Boosting Vehicle-to-Cloud Communication by Machine Learning-Enabled Context Prediction**](https://arxiv.org/abs/1904.10186), In *IEEE Transactions on Intelligent Transportation Systems*, 2019
- B. Sliwa, T. Liebig, R. Falkenberg, J. Pillmann, C. Wietfeld, [**Machine Learning Based Context-Predictive Car-to-Cloud Communication Using Multi-Layer Connectivity Maps for Upcoming 5G Networks**](https://arxiv.org/abs/1805.06603), In *2019 IEEE 88th Vehicular Technology Conference (VTC-Fall)*, 2019
- B. Sliwa, T. Liebig, R. Falkenberg, J. Pillmann, C. Wietfeld, [**Efficient Machine-type Communication using Multi-metric Context-awareness for Cars used as Mobile Sensors in Upcoming 5G Networks**](https://arxiv.org/abs/1801.03290), In *2019 IEEE 87th Vehicular Technology Conference (VTC-Spring)*, 2019, *Best Student Paper*
