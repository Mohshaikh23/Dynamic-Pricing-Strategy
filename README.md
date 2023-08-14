# Dynamic Pricing Model App

This is a Streamlit web application that implements a dynamic pricing model for ride-sharing services. The app allows users to input various parameters and get a predicted ride price based on the trained Random Forest Regressor model. Additionally, it provides visualizations of the model's predictions compared to actual values.

## Table of Contents

- [Dynamic Pricing Model App](#dynamic-pricing-model-app)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)

## Description

The Dynamic Pricing Model App is built using Streamlit, a Python library for creating interactive web applications. The app uses a Random Forest Regressor model trained on historical ride data to predict ride prices based on user input.

## Features

- Predict ride prices based on user inputs such as number of riders, number of drivers, vehicle type, and expected ride duration.
- Visualize the predicted ride prices against actual values using Plotly graphs.
- Provide insights into the distribution of profitable and loss rides.
- Handle missing data through preprocessing techniques.
- Incorporate a Random Forest Regressor model to predict prices.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Mohshaikh23/Dynamic-Pricing-Strategy.git
```

2. Navigate to the project directory:

```bash
cd dynamic-pricing-app
```

3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage
Place your data files (dynamic_pricing.csv, test_data.csv) in the project directory.

Run the Streamlit app:

```bash
streamlit run app.py

```

The app will open in your default web browser. Use the sliders and input fields to adjust the parameters and get the predicted ride price.
Screenshots
Include any relevant screenshots or GIFs of your app in action here.

License
This project is licensed under the MIT License.

