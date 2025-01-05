# HADE-LS
Hierarchical Adaptive Differential Evolution with Local Search for Extreme Learning Machine

## Abstract 
This paper proposes a Hierarchical Adaptive Differential Evolution with Local Search (HADE-LS). In each iteration, HADE-LS partitions the population into four layers based on the fitness value: superior, borderline superior, borderline inferior, and inferior layer. Motivated by the hypothesis from the Proximate Optimality Principle (POP), HADE-LS implements a local search operator for individuals in the superior layer while individuals in other layers utilize the DE/cur-to-pbest/1 mutation and the binomial crossover operator to generate offspring individuals. Each layer independently calibrates its scaling factor and crossover rate based on historical performance metrics. In numerical experiments, we evaluate the performance of HADE-LS on IEEE-CEC2013 and extreme learning machine training tasks against the other nine famous Evolutionary Algorithms (EAs). The experimental results and statistical analysis confirm the competitiveness of HADE-LS.

## Citation
@InProceedings{Zhong:24,  
author="Zhong Rui and Cao Yang and Yu Jun and Munetomo Masaharu",  
editor="Tan, Ying and Shi, Yuhui",  
title="Hierarchical Adaptive Differential Evolution with Local Search for Extreme Learning Machine",  
booktitle="Advances in Swarm Intelligence",  
year="2024",  
publisher="Springer Nature Singapore",  
address="Singapore",  
pages="235--246",  
isbn="978-981-97-7181-3"  
}

## Datasets and Libraries
CEC benchmarks are provided by the opfunu library, while the ELM model and datasets in classification are provided by mafese and intelelm libraries.

