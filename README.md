# User Dashboard Analysis with Machine Learning

## Overview

This repository contains code for a user dashboard analysis tool utilizing machine learning techniques. The tool processes user data  through a google form to evaluate the importance of each factor in predicting specific activities. It employs the k-Nearest Neighbors (kNN) algorithm for classification and compares different scenarios, such as excluding specific sensor data or using alternative labeling approaches.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Evaluation 1: User Importance](#evaluation-1-sensor-importance)
  - [Evaluation 2: Labeling Scenarios](#evaluation-2-labeling-scenarios)
  - [Evaluation 3: Algorithm Comparison](#evaluation-3-algorithm-comparison)
- [Code Structure](#code-structure)
- [Results](#results)

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required Python packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rajdeep432/Analysing-user-behavior.git
   cd user-dashboard-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. Place your sensor data in a CSV file named `allFeatures.csv` in the `data/Test021bis008` directory.

2. Adjust the configuration in the `configuration.py` file if needed.

### Evaluation 1: Sensor Importance

Uncomment the relevant section in the `mainScript` function of `user_importance.py` based on the scenario you want to evaluate. Run the script:

```bash
python sensor_importance.py
```

### Evaluation 2: Labeling Scenarios

Uncomment the relevant section in the `mainScript` function of `labeling_scenarios.py` based on the labeling scenario you want to evaluate. Run the script:

```bash
python labeling_scenarios.py
```

### Evaluation 3: Algorithm Comparison

Uncomment the relevant section in the `mainScript` function of `algorithm_comparison.py` based on the machine learning algorithm you want to evaluate. Run the script:

```bash
python algorithm_comparison.py
```

## Code Structure

The codebase is structured as follows:

- `sensor_importance.py`: Evaluates the importance of each user in predicting activities.
- `labeling_scenarios.py`: Compares different labeling scenarios.
- `algorithm_comparison.py`: Compares different machine learning algorithms.
- `configuration.py`: Configuration settings.
- `data/`: Directory for storing survey data encoded.
- `results/`: Directory for storing evaluation results.

## Results

Results of each evaluation will be printed to the console, providing accuracy scores and confusion matrices.


