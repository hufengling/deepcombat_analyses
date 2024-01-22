#General
library(tidyverse)
library(reshape2)
library(matlib)
library(matrixStats)
library(forcats)

#DeepComBat
library(torch)
library(torchvision)
library(neuroCombat)
library(CovBat)

#ML
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(MASS)

#Statistics
library(twosamples)
library(groupdata2)
library(kBET)

#Visualizations
library(umap)
library(metan)
library(rdist)
library(FactoMineR)

#Plots
library(ggfortify)
library(ggcorrplot)
library(cluster)
library(ggpubr)
library(RColorBrewer)
library(latex2exp)
library(scales)

#Tables
library(gt)
library(gtsummary)

library(here)

#Custom
source("/home/fengling/Documents/nnbatch/code/utils.R")
source("/home/fengling/Documents/nnbatch/code/data_loader.R")
source("/home/fengling/Documents/nnbatch/code/combat_module.R")
#source("/home/fengling/Documents/nnbatch/code/covbat.R")
source("/home/fengling/Documents/nnbatch/code/train.R")
source("/home/fengling/Documents/nnbatch/code/eval_utils.R")
