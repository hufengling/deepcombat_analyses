# Analysis code for DeepComBat manuscript

This is all code used for development and analysis of the DeepComBat image harmonization method described in more detail here: https://www.biorxiv.org/content/10.1101/2023.04.24.537396v1

In the development process, code/deep_combat.Rmd was used to run DeepComBat, along with packages and R scripts of helper functions that are referenced in code/load_packages. Analysis, including Tables and Figures, was performed using the code/deepcombat_analysis.Rmd file. dcVAE and gcVAE were run using code/analyze_output_gcVAE_AL.ipynb, adapted from An et al. (2022).

DeepComBat code has been cleaned up and compiled into an easily-installable package, which can be found here: https://github.com/hufengling/DeepComBat

If useful, please cite: 
Hu, F., Lucas, A., Chen, A.A., Coleman, K., Horng, H., Ng, R.W.S., Tustison, N.J., Davis, K.A., Shou, H., Li, M., Shinohara, R.T., Initiative, T.A.D.N., 2023. DeepComBat: A Statistically Motivated, Hyperparameter-Robust, Deep Learning Approach to Harmonization of Neuroimaging Data. https://doi.org/10.1101/2023.04.24.537396

Bibliography:
An, L., Chen, J., Chen, P., Zhang, C., He, T., Chen, C., Zhou, J.H., Yeo, B.T.T., 2022. Goal-specific brain MRI harmonization. NeuroImage 263, 119570. https://doi.org/10.1016/j.neuroimage.2022.119570
