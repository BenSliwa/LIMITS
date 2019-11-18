Examples
========

**This is a work in process section: Detailed comments are added step by step**

The following examples aim to provide comprehensible guides for applying specific data analysis methods with LIMITS. Each example is concluded by a typical result visualization. Please note that the presented code snippets focus on the core methods of the examples, the full source code can be accessed via the provided links.


## Related Publications

The considered examples are based on scientific analyses which have been carried out in previous publications. Users interested in a more detailed description about the involved challenges and methodological aspects are forwarded to:
- B.Sliwa, C. Wietfeld, [**Empirical Analysis of Client-based Network Quality Prediction in Vehicular Multi-MNO Networks**](https://arxiv.org/abs/1904.10177), In *2019 IEEE 90th Vehicular Technology Conference (VTC-Fall)*, 2019

## Overview

##### Performance Analysis
- [Performance Comparison of Prediction Models](#performance-comparison-of-prediction-models)
- [Model Reapplication and Error Visualization](#model-reapplication-and-error-visualization)
- [Artificial Neural Network Sweet Spot Determination](#artificial-neural-network-sweet-spot-determination)
- [Random Forest Sweet Spot Determination](#random-forest-sweet-spot-determination)
- [Model Convergence Analysis](#model-convergence-analysis)

##### Feature Importance
- [Feature Correlation Analysis](#feature-correlation-analysis)
- [Artificial Neural Network Feature Importance](#artificial-neural-network-feature-importance)
- [Random Forest Feature Importance](#random-forest-feature-importance)
- [Support Vector Machine Feature Importance](#support-vector-machine-feature-importance)
- [Feature Reduction](#feature_reduction)

##### Model Visualization
- [Artificial Neural Network Visualization](#artificial-neural-network-visualization)
- [Random Forest Visualization](#random-forest-visualization)


## Performance Analysis

### Performance Comparison of Prediction Models

[[Complete source code of the example]](src/example_experiment.py)

In this example, we apply multiple supervised machine learning methods in order to forecast the achievable data rate based on measured passive network quality indicators which are provided in the data set *mnoA.csv*. Therefore, we set up an **experiment** which sequentially performs a **cross validation** of each prediction model.

```python
training = "../examples/mnoA.csv"
models = [ANN(), M5(), RandomForest(), SVM()]
e = Experiment(training, "example_experiment")
e.regression(models, 10)
```

For each of the 10 cross validation runs, temporary data is stored in the */tmp/* folder.

```python
files = ["tmp/cv_" + str(i) + ".csv" for i in range(len(models))]
fig, axs = plt.subplots(2,2)
fig.set_size_inches(8, 5)
xticks = ["ANN", "M5", "Random Forest", "SVM"]
ResultVisualizer().boxplots(files, "r2", xticks,  ylabel='R2', fig=fig, ax=axs[0][0], show=False)
ResultVisualizer().boxplots(files, "mae", xticks,  ylabel='MAE [MBit/s]', fig=fig, ax=axs[0][1], show=False)
ResultVisualizer().boxplots(files, "rmse", xticks,  ylabel='RMSE [MBit/s]', fig=fig, ax=axs[1][0], show=False)
ResultVisualizer().boxplots(files, "training", xticks,  ylabel='Training Time [s]', fig=fig, ax=axs[1][1], savePNG="results/" + e.id + "/"+'example_experiment.png')
```

![example_experiment](misc/example_experiment.png)


### Model Reapplication and Error Visualization

[[Complete source code of the example]](src/example_model_reapplication.py)

For each cross validation run, a prediction model is learned and a *C++* implementation of the trained model is exported. The **CodeEvaluator** module then compiles a dummy version of the model and replays all measurements contained in the *test set* of the current fold.

```python
ce = CodeEvaluator()
R = ResultMatrix()
for i in range(10):
	codeFile = resultFolder + "rf_disc.cpp"
	data = "\n".join(FileHandler().read("tmp/raw0_" + str(i) + ".txt"))
	model.exportCode(data, csv, attributes, codeFile)

	resultFile = resultFolder + "scatter_" + str(i) + ".txt"
	keys, res, conf = ce.regression(codeFile, csv.findAttributes(0), "tmp/test_mnoA_" + str(i) + ".csv", resultFile)
	R.add(keys, res)

ResultVisualizer().scatter([resultFolder + "scatter_" + str(i) + ".txt" for i in range(10)], "prediction", "label", xlabel='Predicted Data Rate [MBit/s]', ylabel='Measured Data Rate [MBit/s', savePNG=resultFolder+'example_model_reapplication.png')
```

![example_correlation](misc/example_model_reapplication.png)



### Artificial Neural Network Sweet Spot Determination

[[Complete source code of the example]](src/example_ann_sweet_spot.py)



![example_rf_sweet_spot](misc/example_ann_sweet_spot.png)



### Random Forest Sweet Spot Determination

[[Complete source code of the example]](src/example_rf_sweet_spot.py)

![example_rf_sweet_spot](misc/example_rf_sweet_spot.png)








### Model Convergence Analysis

[[Complete source code of the example]](src/example_model_convergence.py)

```python
resultFile = "tmp/convergence_rf.txt"
ConvergenceAnalysis().run("../examples/mnoA.csv", M5(), 100, resultFile)
ResultVisualizer().errorbars([resultFile], "r2")
```

![example_model_convergence](misc/example_model_convergence.png)



## Feature Analysis


### Feature Correlation Analysis

[[Complete source code of the example]](src/example_correlation.py)

```python
resultFolder = "results/example_correlation/"
resultFile = resultFolder + "corr.csv"

csv.computeCorrelationMatrix(resultFile)
ResultVisualizer().colorMap(resultFile, savePNG=resultFolder+'example_correlation.png')
```


![example_correlation](misc/example_correlation.png)


### Artificial Neural Network Feature Feature Importance

[[Complete source code of the example]](src/example_ann_feature_importance.py)


![example_ann_feature_importance](misc/example_ann_feature_importance.png)



### Random Forest Feature Importance

[[Complete source code of the example]](src/example_rf_mdi.py)

```python
M = CSV("tmp/features_0.csv").toMatrix()
M.normalizeRows()
M.sortByMean()
M.save("tmp/rf_features.csv")
```

```python
ResultVisualizer().barchart("tmp/rf_features.csv", xlabel="Feature", ylabel="Relative Feature Importance", savePNG=e.id+".png")
```

![example_rf_mdi](misc/example_rf_mdi.png)


### Support Vector Machine Feature Importance

[[Complete source code of the example]](src/example_svm_feature_importance.py)


![example_svm_feature_importance.py](misc/example_svm_feature_importance.py.png)


### Feature Reduction

[[Complete source code of the example]](src/example_feature_reduction.py)

```python
M = CSV("tmp/features_0.csv").toMatrix()
M.normalizeRows()
M.sortByMean()
```
```python
for i in range(len(M.header)-1):
	key = M.header[-1]
	M.header = M.header[0:-1]
	csv.removeColumnWithKey(key)
	csv.save(subset)

	e = Experiment(subset, "example_feature_reduction")
	e.regression([model], 10)
```

![example_feature_reduction](misc/example_feature_reduction.png)



## Model Visualization

### Artificial Neural Network Visualization

[[Complete source code of the example]](src/example_ann_visualization.py)

```python
training = "../examples/mnoA.csv"
model = ANN()
model.hiddenLayers = [10, 10]

e = Experiment(training, "example_ann")
e.regression([model], 10)
```

```python
data = "\n".join(FileHandler().read("tmp/raw0_0.txt"))
annModel = model.generateClassificationModel(data, csv.findAttributes(0), model.hiddenLayers, training)
annModel.exportEps('ann_vis.eps')
```

![example_rf_visualization](misc/example_ann_visualization.png)




### Random Forest Visualization

[[Complete source code of the example]](src/example_rf_visualization.py)

```python
training = "../examples/vehicleClassification.csv"
model = RandomForest()
model.depth = 7

e = Experiment(training, "example_rf")
e.classification([model], 10)
rf.exportEps(model.depth+1, 10, 10, len(attributes)-1)
```

```python
attributes = csv.findAttributes(0)

data = "\n".join(FileHandler().read("tmp/raw0_0.txt"))
rf = model.generateModel(data, attributes)
rf.exportEps(model.depth+1, 10, 10, len(attributes)-1)
```

![example_rf_visualization](misc/example_rf_visualization.png)
