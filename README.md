# BTC Directional Prediction Pipeline

## Overview
A complete machine learning pipeline that predicts Bitcoin price direction using 2 years of hourly OHLCV data from Binance.

## Results
- Overall Accuracy: 0.5673
- Non-flat Directional Accuracy: 0.5676
- High Confidence Accuracy: 0.5797
- Majority Class Baseline: ~0.50

## Setup
pip install -r requirements.txt

## Run
python download.py
python train.py
python evaluate.py

## Features
- log_return: measures hourly price change
- volatility: measures market turbulence
- rsi: measures overbought/oversold conditions
- bb_deviation: measures price distance from mean
- macd: measures momentum direction
- volume_zscore: measures unusual volume activity
- taker_buy_ratio: measures buying vs selling pressure

## Model Performance
- LightGBM Average CV Accuracy: 0.5196
- Logistic Regression Average CV Accuracy: 0.5406