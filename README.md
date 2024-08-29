# Prometheus code
## Overview ##
Prometheus is a Python tool designed for analyzing and processing experimental data related to radiance, temperature, and emissivity. This tool provides functionalities for reading, smoothing, interpolating, and binning data, as well as generating plots and exporting results to CSV files.

## Features ##

-Read and process radiance, temperature, and emissivity data from text files.

-Smooth and interpolate data.

-Bin data based on particle sizes.

-Generate and save plots.

-Export processed and binned data to CSV files.

## Prerequisites ##

Before using Prometheus, ensure you have the following:

-Python 3.x

-Required Python libraries:

    -numpy
    
    -pandas
    
    -matplotlib
    
    -statsmodels
    
    -scikit-learn
    
    -scipy

## Usage ##

### 1. Import the Prometheus Class ###

  Import the Prometheus class into your Python script or interactive environment:
    
    from prometheus import Prometheus

### 2. Create an Instance of the Prometheus Class ###

  Instantiate the Prometheus class:
      
      p = Prometheus()
      
  You will be prompted to provide:
  
      -Address of folder with PMT txt files: Enter the path to the folder containing radiance, temperature, and emissivity text files.
      -Sample Name: Enter a name for the sample, which will be used in output file names.

### 3. Run the Data Analysis ###

  Call the get_fire method to process the data:
    
    p.get_fire()
    
  This method will:
    
      -Prompt for particle sizes if available. (You will have to enter size of partilce/ R.O.I for each run based on run identification
      
      -Process the radiance, temperature, and emissivity data.
      
      -Perform smoothing and interpolation.
      
      -Bin data based on particle sizes (if provided).
      
      -Generate and save plots (if specified).
      
      -Export processed and binned data to CSV files.
  
### Optional Settings ###

You can adjust the following settings in the Prometheus class:

    -run_plot_choice: Set to 'y' to generate plots of the normalized dataset, 'n' to skip plotting.
    
    -iterative_plot_choice: Set to 'y' to visualize iterative fitting for temperature data.
    
    -error_confidence: Set to 'y' to filter out temperature data with high error confidence.
    
    -binned_plots_choice: Set to 'y' to generate plots for binned data, 'n' to skip plotting.


