# Time Series Decomposition Project

Model code and figures for the manuscript _Time Series Decomposition of Continuous Flow Cytometry Reveals Biogeographical Variation in Population-Specific Productivity of Small Phytoplankton_. 

## Overview
Files to run primary production TSD model for flow cytometry time series application. This model uses simulated datasets, [a published laboratory experiment](https://doi.org/10.1371/journal.pone.0005135), and _in situ_ times series to generate estimates of population-specific productivity.

## Authors
Katherine L. Qi, Francois Ribalet, Sangwon Hyun, Angelicque E. White, E.
Virginia Armbrust

## Base Requirements
Python 3.8, Anaconda

## Installation
Clone this repository into your current working directory:

``` 
git clone git@github.com:klqi/seaflow_tsd.git
```

Create a conda environment from this repository:

```
conda env create -f environment.yml
```

Or, use an existing, activated conda environment:

```
conda env update -f environment.yml
```
