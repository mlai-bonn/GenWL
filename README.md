# GenWL
A Generalized Weisfeiler-Lehman Graph Kernel

## Dependencies
* numpy==1.19.1
* networkx==2.2
* POT==0.7.0
* scikit_learn==0.24.1

## Run the code (Example)
The following command runs the script on the MUTAG dataset using the approximation variant: \
`<python main.py Data/MUTAG --approx>`

Additional arguments are as follows:
```
positional arguments:
  dataset     Path to dataset

optional arguments:
  -h, --help  show this help message and exit
  --approx    Use approximation variant
  --h H       Max. unfolding tree depth
  --c C       Number of paritionings
```
