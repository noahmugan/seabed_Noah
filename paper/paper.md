---
title: 'SEABED: SEquential Analysis and Bayesian Experimental Design powered by JAX'
tags:
  - Python
  - Bayesian inference
  - Experimental design
  - Particle filtering
authors:
  - name: Paul Kairys
    orcid: 0000-0002-9203-9513
    email: kairyspm@ornl.gov
    affiliation: "1,2"
  - name: Noah Mugan
    affiliation: 3
  - name: Daniel P. Mark
    affiliation: 3
  - name: Atlas Bailey
    affiliation: 3
  - name: Jonathan C. Marcks
    affiliation: "2, 3, 4" 
  - name: Nazar Delegan
    affiliation: "3, 2"
  - name: Jiefei Zhang
    affiliation: "3, 2"
  - name: F. Joseph Heremans
    affiliation: "3, 2, 4" 
affiliations:
 - name: Mathematics and Computer Science Division, Argonne National Laboratory, Argonne, IL 60439 USA
   index: 1
 - name: Materials Science Division, Argonne National Laboratory, Argonne, IL 60439 USA
   index: 3
 - name: Q-NEXT, Argonne National Laboratory, Argonne, IL 60439 USA
   index: 2
 - name: Pritzker School of Molecular Engineering, University of Chicago, Chicago, IL 60637 USA
   index: 4
date: 5 August 2024
bibliography: paper.bib
---

# Summary

SEquential Analysis and Bayesian Experimental Design (SEABED) is a Python 3 package designed to implement efficient sequential Bayesian inference and experimental design in an application agnostic manner. Leveraging a particle-based representations of probability density functions, SEABED is a modular, scalable, and customizable library for Bayesian inference and experimental design. In practice, this allows users to approximate the parameters of unknown models in their respective domains while controlling and designing the frequency or complexity of experiments. Importantly, SEABED’s particle filtering approach enables efficient parallelization of calculations via the JAX package, a library for just-in-time (JIT) compilation of array-based computation. Sequential updates to the particle probability distribution also allow this package to adaptively select input settings with the highest utility, thereby gaining maximal information with each measurement and further improving experimental efficiency. 

SEABED’s broad functionality makes it widely applicable to a variety of Python-based projects across scientific domains. The JAX ecosystem lets users easily implement this package in projects involving deep learning or solving differential equations, and the modularity of SEABED enables users to tailor utility functions to a project's individual needs. Since its release in 2023, SEABED has already found success in accelerating the adaptive characterization of solid-state quantum defects [@autodisc_paper] and shows promise for future applications across domains. 

# Statement of need

Within the field of statistical analysis, Bayesian inference is a fundamental process used to identify the underlying, typically fundamental, parameters of a system that may not be directly observable. After defining a parametric probability distribution and an associated likelihood function, a Bayesian model can determine the likelihoods of different observed data and iteratively estimate underlying parameters from observations. However, one may also wish to identify what types of system parameters can be modified in order for the observed data to provide as much information as possible on the parameters of interest, this is a task typically referred to as experimental design. These experiments are typically selected using a utility measure. Experimental settings with higher utility are more likely to reveal useful information and can filter out unlikely parameters than measurements with lower utility.

Existing software packages such as PyMC [@pymc_paper], Pyro [@pyro_paper], TensorFlow Probability [@tensorflow_paper], Turing.jl [@turing_paper], or Blackjax [@blackjax_paper] that enable Bayesian inference do not support or emphasize experimental design. These existing packages also emphasize implementing Bayesian inference using either Markov chain Monte Carlo (MCMC) [@mcmc_book] or Hamiltonian Monte Carlo (HMC) [@hmc_paper] methods. These methods are very robust and flexible, however are not as well suited for practical experimental design, where experimental settings must often be chosen in near real time, making particle filtering methods particularly attractive [@particle_paper]. Additionally, many of these packages also define probabilistic programming languages (PPL). PPLs provide a high-level interface which allow for users to build and sample from complex statistical models and are often tailored to a particular application or domain. In contrast, SEABED does not implement a PPL and instead provides a streamlined approach to particle-filtering-based inference of arbitrary statistical models and software. This allows increased user customization and greater flexibility on the numerical building blocks. 

SEABED began as a fork of OptBayesExpt [@optbayesexpt_paper] but generalizes and streamlines the numerical aspects of adaptive Bayesian experimental design while maintaining JAX compatibility. This compatibility enables the utilization of GPUs and TPUs in calculations making it particularly valuable for statistical models with many parameters. SEABED also includes functions which update the particle distribution based on a collection of measurements as opposed to a single data point, known as "batch processing," which increases the computational stability of particle-based methods.

In general, SEABED provides a software framework for Bayesian inference and experimental design with particle filtering and provides new capabilities for adaptive experimentation, numerical analysis, and scientific computing in a way that was not supported by the existing scientific software ecosystem. 

# Functionality

\autoref{fig:flowchart} visualizes the typical SEABED inference process. Following the definition of a likelihood function and creation of the particle prior, the package can begin to select optimal measurements and update its particle distribution accordingly. Each Bayesian update can be completed by considering either a single data point or multiple, with multiple points increasing the accuracy and stability of particle filtering. Software utilizing SEABED can also cycle between Bayesian inference and experimental design exploration. This allows users to choose experimental settings quantitatively and most informative measurements.


![A flowchart for typical SEABED implementation. Scripts can cycle between measurements and particle resampling up to a predefined end point.\label{fig:flowchart}](flowchart.png)

The utility calculation included in the SEABED package is one based on Shannon entropy [@shannon_paper]. This is a method of calculating the information for a random variable with a given probability distribution. For an input $s$, output $y$, and particle distribution $A$, the particle's Shannon entropy is calculated as

\begin{equation}
    H(s, y, A) = -\sum_{\vec{\alpha}\in A}P(y|s, \vec{\alpha})\log[P(y|s, \vec{\alpha})],
\end{equation}
where $P(y|s, \vec{\alpha})$ is the probability of observing $y$ given input $s$ and particle $\vec{\alpha}$. 

For a given output, the utility measure computes the expected change in Shannon entropy after resampling particles. Because high utility measurements will lead to a posterior with low entropy, the utility is returned as -1 times the entropy change. The utility for each input is then determined by considering the utility measure averaged over each possible output. 

The entropy-based utility calculation is widely applicable, and thus included as the default utility metric for SEABED. The package also easily accepts a user-defined utility function for contexts which require a more unique approach [@simple_utilities]. 

# Acknowledgements
SEABED originated as a fork of OptBayesExpt [@optbayesexpt_paper] and has since undergone significant deviations from the original software and structure. This work is supported by the Laboratory Directed Research and Development program of Argonne National Laboratory, as part of the Autonomous Discovery Initiative. We acknowledge additional support in part by the U.S. Department of Energy, Office of Science, Office of Workforce Development for Teachers and Scientists (WDTS) under the Science Undergraduate Laboratory Internships Program (N.M.), and the Q-NEXT Quantum Center supported by the U.S. Department of Energy, Office of Science, National Quantum Information Science Research Centers (J.C.M.), including through the Open Quantum Initiative (A.B.). 

# References