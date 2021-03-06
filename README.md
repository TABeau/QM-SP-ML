# QM-SP-ML
Molecular design using signal processing and machine learning

## GOALS

Accumulation of molecular data obtained from quantum mechanics (QM) theories such as density functional theory (DFTQM) make it possible for machine learning (ML) to accelerate the discovery of new molecules, drugs, and materials. Models that combine QM with ML (QM↔ML) have been very effective in delivering the precision of QM at the high speed of ML.

In this project, we are integrating well-known signal processing (SP) techniques (i.e. short time Fourier transform, continuous wavelet analysis, Wigner-Ville distribution, etc.) in the QM↔ML pipeline. The new QM↔SP↔ML pipeline represents a powerful machinery that can be used for representation, visualization, forward and inverse design of molecules.

## VISUALIZATION 

The time-frequency-like (TFL) representation of molecules encodes their structural, geometric, energetic, electronic and thermodynamic properties.

## FORWARD DESIGN 

The TFL representation is used in the forward design loop as input to a deep convolutional neural networks trained on DFTQM calculations, which outputs the properties of the molecules. Tested on the QM9 dataset (composed of 133,855 molecules and 19 properties), the new QM↔SP↔ML model is able to predict the properties of molecules with a mean absolute error (MAE) below acceptable chemical accuracy (i.e. MAE < 1 Kcal/mol for total energies and MAE < 0.1 ev for orbital energies). Furthermore, the new approach performs similarly or better compared to other ML state-of-the-art techniques described in the literature.  See our paper at: https://arxiv.org/abs/2004.10091

## INVERSE DESIGN 

In Progress stay tuned

## CONCLUSION 

In all, in this project, preliminary results show that the QM↔SP↔ML model represents a powerful technique for molecular representation, visualization, forward and inverse design.

## REFERENCES

1- A. B. Tchagang and J. J. Valdes, “Discrete Fourier Transform Improves the Prediction of the Electronic Properties of Molecules in Quantum Machine Learning,” 2019 IEEE Canadian Conference of Electrical and Computer Engineering (CCECE), 2019.

2- A. B. Tchagang, J. J. Valdes, and A. H. Tewfik "Molecular Design Using Signal Processing and Machine Learning: Time-Frequency-like Representation and Forward Design", arXiv:2004.10091 [physics.chem-ph], 2020, https://arxiv.org/abs/2004.10091, Technical Report.

3- A. B. Tchagang and J. J. Valdes "Time Frequency Representations and Deep Convolutional Neural Networks: A Receipe for Molecualar Forward Design", Submitted Under Review ICASSP 2021.

4- A. B. Tchagang and J. J. Valdes "Time Frequency Representations and Deep Generative Models: A Receipe for Molecualar Inverse Design", Under Preparation 2021.

5- A. B. Tchagang, et al., "Molecular Design Using Signal Processing and Machine Learning: Inverse Design", Under Preparation, 2021.
