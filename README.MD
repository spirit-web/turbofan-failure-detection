# Turbofan Failure Detection


Machine learning project using the **NASA C-MAPSS Turbofan Engine Degradation Dataset** to investigate whether engine degradation can be detected before failure.


## Project Question


> Can we detect when a turbofan engine is beginning to fail before the actual failure occurs?


The project compares two deep learning approaches for detecting changes in engine sensor data:


- **TCN (Temporal Convolutional Network)** – predicts upcoming sensor behaviour based on previous engine cycles. Increasing prediction error may indicate that the engine behaviour is changing.
- **Autoencoder** – learns patterns associated with healthy engine operation. Increasing reconstruction error may indicate that the engine is behaving differently from the healthy patterns learned by the model.


The goal is not to directly compare the raw error values of the two models, but to evaluate how well their warning signals relate to actual engine degradation.


## Dataset


**NASA C-MAPSS Turbofan Engine Degradation Dataset**
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps


The project uses the **FD001** subset:


- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`


FD001 contains one operating condition and one fault mode.


The dataset consists of simulated run-to-failure data from turbofan engines. Each engine is monitored through multiple sensor measurements over a sequence of operating cycles.


## Method


To make the comparison between the models as consistent as possible:


- Both models use the same sensor selection.
- Approximately constant sensors are removed.
- Both models use the same sequence/window length.
- A window size of approximately **30 cycles** is used.
- The first approximately **30% of each training engine's lifetime** is considered healthy operation.
- Both models are trained using healthy engine data.


The models are then evaluated on later engine cycles that were not included in the healthy training data.


## Evaluation


The model warning signals are evaluated against the dataset's **Remaining Useful Life (RUL)**.


Two main evaluation methods are used:


1. **Correlation with RUL**  
   Measures how strongly the model's warning signal relates to decreasing remaining engine life.


2. **Early Warning Time**  
   Measures how many cycles before failure the model's warning signal crosses a defined anomaly threshold.


## Project Structure


```text
turbofan-failure-detection/
│
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── tcn.ipynb
│   ├── autoencoder.ipynb
│   └── analysis.ipynb
├── reports/
│   └── figures/
├── .gitignore
├── README.md
└── requirements.txt
```


## Models


### TCN
Temporal modelling of engine sensor sequences to detect unexpected changes in future sensor behaviour.


### Autoencoder
Reconstruction-based anomaly detection where deviations from learned healthy engine behaviour produce increasing reconstruction errors.


## Team


- **Emmy**
- **Mahtab**
- **Spirit**


## Status


Work in progress.