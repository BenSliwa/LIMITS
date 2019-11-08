Examples
========

**TODO** ADD DETAILED DESCRIPTIONS


- [Performance Comparison of Prediction Models](#performance-comparison-of-prediction-models)
- [Model Reapplication and Error Visualization](#model-reapplication-and-error-visualization)
- [Random Forest Sweet Spot Determination](#random-forest-sweet-spot-determination)
- [Feature Correlation Analysis](#feature-correlation-analysis)
- [Model Convergence Analysis](#model-convergence-analysis)
- [Artificial Neural Network Visualization](#artificial-neural-network-visualization)
- [Random Forest Visualization](#random-forest-visualization)



### Performance Comparison of Prediction Models

[[Complete source code of the example]](src/example_experiment.py)

```python
training = "../examples/mnoA.csv"
models = [ANN(), M5(), RandomForest(), SVM()]

e = Experiment(training, "example_experimment")
e.regression(models, 10)
resultFolder = "results/" + e.id + "/"

ResultVisualizer().barChart(resultFolder+"result.csv", "r2", ['ANN', 'M5', 'Random Forest', 'SVM'], ylabel='R2', savePNG=resultFolder+'example_experimment.png')
```

![example_experiment](misc/example_experiment.png)


### Model Reapplication and Error Visualization

[[Complete source code of the example]](src/example_model_reapplication.py)

```python
CodeEvaluator().regression(codeFile, csv.findAttributes(0), training, resultFile)
ResultVisualizer().scatter(resultFile, "prediction", "label", xlabel='Predicted Data Rate [MBit/s]', ylabel='Measured Data Rate [MBit/s', savePNG=resultFolder+'example_model_reapplication.png')
```

![example_correlation](misc/example_model_reapplication.png)


### Random Forest Sweet Spot Determination

[[Complete source code of the example]](src/example_rf_sweet_spot.py)

![example_rf_sweet_spot](misc/example_rf_sweet_spot.png)


### Feature Correlation Analysis

[[Complete source code of the example]](src/example_correlation.py)

```python
resultFolder = "results/example_correlation/"
resultFile = resultFolder + "corr.csv"

csv.computeCorrelationMatrix(resultFile)
ResultVisualizer().colorMap(resultFile, savePNG=resultFolder+'example_correlation.png')
```


![example_correlation](misc/example_correlation.png)


### Model Convergence Analysis

[[Complete source code of the example]](src/example_model_convergence.py)

```python
resultFile = "tmp/convergence_rf.txt"
ConvergenceAnalysis().run("../examples/mnoA.csv", M5(), 100, resultFile)
ResultVisualizer().errorbars([resultFile], "r2")
```

![example_model_convergence](misc/example_model_convergence.png)

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
