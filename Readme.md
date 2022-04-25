# Evolutionary Counterfactual Visual Explanation

## Run 
`python Evo_Patch_Level.py [TrainingData][NumPatches][Target][Dataset][model][nbr_classes][problem][epochs][population_size][mig_rate][cosover][cxpb][mutation][mutpb][algorithm][save] [sim]`

Example: 
`python Evo_Patch_Level.py ShufflePatches 14 0 mnist ./Model/cnn_MNIST.h5 10 Basic 500 1000 0 uniform auto2 randomAugmentedPatch auto2 NSGA2 ./Results/MNIST/`
## Run the Experiments
### Distances
`python Experiment1.py [dataset]`
### Mutation
`python Experiment2.py [dataset]`
### Results compared with Benchmark 
`python Experiment3.py [dataset]`
Run Code in Folder Alibi_Benchmark. (To run the Alibi Benchmark a different environment is necessary as Alibi runs on tensorflow 1 while we use tensorflow 2. `pip intall -r ./Alibi_Benchmark/requirments.txt`)
Run  `python ./Results/Compare_with_Benchmark.py`
## Customization
### Add your own Problem
1. Write your problem with `pymop`
2. Add your problem to the folder  `MultiObjectiveProblem`
3. Add a reference to the problem in the function dict of `Evo_Patch_Level.py`
4. Run via commandline using the dict key given in setp 3. 
### Add your own Dataset
1. Write a dataloader for your data in  `Additional_Help_Functions.py`
2. Add a reference to the data loader in the function dict of `Evo_Patch_Level.py`
3. Run via commandline using the dict key given in setp 2.
### Add your own Classification Model
1. Put .h5 into the folder  `Model`
2. Run cmd with model path