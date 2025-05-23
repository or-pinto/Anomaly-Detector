# 🤖 Robot Anomaly Detection

A machine learning project for detecting anomalies in robot sensor data using differential feature engineering and neural networks.

## 📝 Overview

This project implements an anomaly detection system for robotic systems by analyzing various sensor measurements including:
- Motor velocities, positions, and torques
- Joint measurements
- System current and voltage readings
- Target values

## 🎯 Key Features

- Differential feature engineering from robot sensor data
- Multi-class anomaly detection
- Support for 6-motor robot configurations
- Detection of various anomaly types (friction, weight, miscommutation)

## 🛠️ Installation

1. Clone the repository
2. Install dependencies from pyproject.toml using Poetry:
```bash
poetry install
```
or alternatively using requirements.txt
```bash
pip install -r requirements.txt
```

## 📊 Data

The project expects sensor data in `.parquet` format with the following measurements:
- Motor measurements (velocity, position, torque, current)
- Joint positions and velocities
- System measurements
- Categorical labels for anomaly types

## 🚀 Usage

1. Place your sensor data file as `data.parquet` in the project directory
2. Run the Jupyter notebook to:
   - Load and process sensor data
   - Generate differential features
   - Train the anomaly detection model
   - Evaluate results

## 📄 License

MIT License.
