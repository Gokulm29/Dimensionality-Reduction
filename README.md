# Dimensionality Reduction Project

This project focuses on applying dimensionality reduction techniques to high-dimensional datasets, a critical step in preprocessing data for machine learning and visualization tasks. The notebook provides a comprehensive implementation and explanation of various dimensionality reduction algorithms and their applications. Additionally, the project incorporates the **Gaussian Naive Bayes (GaussianNB)** classifier to analyze the effectiveness of dimensionality reduction techniques in predictive modeling. The project also includes exploratory data analysis (EDA), metrics evaluation, and timing analysis to measure the performance and efficiency of the methods.

## Project Overview

Dimensionality reduction is essential for simplifying complex datasets while preserving significant patterns and structures. This project explores:

- **Principal Component Analysis (PCA):** A linear technique that transforms the data into a lower-dimensional space while retaining the maximum variance.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE):** A non-linear method primarily used for visualization.
- **Uniform Manifold Approximation and Projection (UMAP):** A non-linear dimensionality reduction method that preserves both local and global structure.

## Features

- Exploratory Data Analysis (EDA) to understand dataset characteristics.
- Comparison of different dimensionality reduction techniques.
- Integration of the **GaussianNB** classifier to evaluate the impact of dimensionality reduction on predictive performance.
- Evaluation of performance metrics such as accuracy, precision, recall, and F1-score.
- Calculation of start and end times for each method to analyze computational efficiency.
- Visualization of reduced dimensions.
- Insights into dataset patterns and structure.

## Technologies Used

- **Python**
- **Google Colab**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Requests**
- **BeautifulSoup**
- **Zipfile**
- **LogisticRegression**

## Libraries Used

Below are the Python libraries used in this project:

```python
import requests
from bs4 import BeautifulSoup
import zipfile
import io
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import time
```

## Getting Started

### Prerequisites

Make sure you have a Google account and access to Google Colab. Install the required libraries directly in the Colab notebook using the following commands:

```python
!pip install numpy pandas scikit-learn matplotlib seaborn requests beautifulsoup4
```

### Dataset

import requests
import zipfile
import io

# Download the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
response = requests.get(url)

# Extract the dataset
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("./data")

print("Dataset downloaded and extracted successfully.")

### Running the 

1. Open the `Dimensionality Reduction.ipynb` file in Google Colab.
2. Upload your dataset as instructed in the notebook.
3. Perform Exploratory Data Analysis (EDA) to understand the dataset.
4. Run each cell sequentially to explore and execute the code.

## Folder Structure
```
.
├── Dimensionality Reduction.ipynb   # Main project notebook
├── data/                            # Folder for datasets (if downloaded locally)
```


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- Inspiration from [Scikit-learn Documentation](https://scikit-learn.org/).
- References to research papers and tutorials on dimensionality reduction techniques.
