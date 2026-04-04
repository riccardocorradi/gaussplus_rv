# Gauss+ Model (Work in Progress)

This repository contains an ongoing implementation of the **Gauss+ affine term structure model (ATSM)**.

The project is currently under development, but here's a brief overview of the content:
- in `/notes` you find a document containing all the steps to justify the formulas used
- in `/pricing` you find code for the pricing engine (`pricer.py`) that is able to build spot zero-coupon rate and forward curves given a set of parameters. We test the model's functioning in pricer tests.ipynb
- in `/sim` you find code to generate paths according to the three SDEs governing the model, which we then use in `simulation tests.ipynb` to check whether the calibration procedure recovers the true model parameters
- in `calibration.py` you find the code for the Calibration() class. The class does everything from calibrating mean reversions, to extracting latent factors, to computing the risk premia priced into the forward curve.
- in `treasury.ipynb` you find the implementation of the model on U.S. Treasury bootstrapped ZCB yields from the NY Fed Dataset

The goal is to calibrate the model following the approach outlined in **Tuckman & Serrat**, ensuring consistency with observed market data.

More details and documentation will be added as the implementation progresses.
