# MLpse

MLpse is a package designed for the Maximum Likelihood Estimation of Power Spectrum of the transit radio data with the m-mode formalism.

To do this, it depends on various related packages including:
- [driftscan](https://github.com/hirax-array/driftscan): for modelling the telescope and generating the products, for example, SVD and KL 
transformation matrices, required for simulation and analysis.
- [draco](https://github.com/hirax-array/draco/tree/master/draco): for simulating the time stream data from maps of the sky and transform 
it to m-mode data
  - [cora](https://github.com/hirax-array/cora): for modelling and simulating the radio sky
  - [driftscan](https://github.com/hirax-array/driftscan): as introduced above.
- [caput](https://github.com/hirax-array/caput): provides infrastructure for building these packages, especially MPI utilities.

## Descriptions of the package:
MLpse consists of three files:
1. "maximum_likelihood.py": This is the major script. It calls modules defined in the other two files. Run this to get an ".hdf5" file which records 
the values of power spectrum parameters and corresponding k bands.
2. "MLpse.py": core source code with 3 classes:
  - *kspace_cartesian*: for parameterizing and binning the k space.
  
  - *Covariances*: for dealing with the covariance matrices and projecting them to KL basis.
  
  - *Likelihood*: for calculating the per-mmode log-likelihood function and the associated Jacobians and Hessian matrics.
  
3. "Fetch_info.py": this is the module file adapted from "[drfitscan/core/manager.py](https://github.com/hirax-array/driftscan/blob/master/drift/core/manager.py)" for reading and translating the parameter file used in driftscan

## Typical workflow of MLpse with simulated data using the m-mode analysis pipeline:
1. Model the telescope and generate Beam transfer matrics, SVD and Karhunen-Loève (KL) transform matrices, etc. The settings 
of the telescope and SVD/KL filters are specified in the parameter file. Run the parameter file with command:
```
drift-makeproducts run product_params.yaml
```
or if you run with mpi
``` 
srun --mpi=pmi2 drift-makeproducts run /home/zzhang/parameter_files/prod_params.yaml
```
Example of the parameter file:
```
config:
    # This tells driftscan what you want it to do
    beamtransfers:      Yes     # Generate beam transfer matrices
    kltransform:        Yes     # Generate KL filter
    psfisher:           Yes     # Generate PSE products

    output_directory:   /data/zzhang/sim1/bt_matrices # Product directory. Beam transfer matrices and associated products will go here.

telescope:
    type:
        # Point to the telescope class we want to use
        class:  HIRAXSurvey
        module: hirax_transfer.core

    # Do 3 pointings of the array spanning declinations +-5 deg. off zenith
    pointing_start: -5
    pointing_stop: 5
    npointings: 3

    hirax_spec:
        #  Specify the frequency channels
        freq_lower: 600.
        freq_upper: 650
        num_freq: 16
        freq_mode: edge         # This means freq_{upper,lower} specify channel edges

        auto_correlations: No   # Don't use single dish data, only interferometer mode.
        tsys_flat: 50.0

        dish_width: 6.0         # In metres, this is only used for the minimum l and m calculation

        # Specify limits on m and l to use
        lmax: 200
        mmax: 200

        # When noise weighting is needed, will assume an integration
        # time of this many days.
        ndays: 733

        # Set the array layout
        hirax_layout: # See hirax_transfer.layouts
            type: square_grid
            spacing: 9.0        # Distance between dish centres in metres
            grid_size: 8        # ie. 3x3 grid


        hirax_beam: # See hirax_transfer.beams
            # Gaussian beam with FWHM (intensity) of fwhm_factor*lambda/diameter
            type: gaussian
            diameter: 6.        # In metres
            fwhm_factor: 1.0

kltransform:
  # See driftscan.drift.core.{kltransform.py,DoubleKL.py}

  # This is a more realistic one. With a larger sim, higher thresholds would be better
    - type: DoubleKL
      name: dk_1thresh_fg_100thresh #  This is a name to give to the filter
      inverse: Yes
      threshold: 1            # Second stage Signal/(FG+noise) filter
      use_thermal: Yes
      use_foregrounds: Yes
      use_polarised: Yes
      foreground_threshold: 100 # First stage Signal/FG filter

```

2. Model and simulate the radio sky, both the cosmological signal and the foregrounds. Examples are
```
cora-makesky --pol full --nside 512 --freq-mode edge --freq 600 650 16 --filename cora_sim_21cm_nside_512.h5 21cm
cora-makesky --pol full --nside 512 --freq-mode edge --freq 600 650 16 --filename cora_sim_galaxy_nside_512.h5 galaxy
```

3. Simulate a sidereal visibility stream, and generate the visibilities in KL basis, from the products generated above. 
```
caput-pipeline run config_draco.yaml
```
An example for the parameter file:
```
        -   type:       draco.core.task.SetMPILogging
  
          # Load the product manager object from our driftscan run
        -   type:       draco.core.io.LoadProductManager
            out:        pm
            params:
              product_directory:  "/data/zzhang/sim1/bt_matrices"

        -   type:       draco.core.io.LoadMaps
            out:        imap
            # Load maps. Each map in this list will be added together. Here we'll sum a 21cm and galactic FG map
            params:
                maps:
                  - files:
                    - "/home/zzhang/cora_sim_21cm_nside_128.h5"
                    - "/home/zzhang/cora_sim_galaxy_nside_128.h5"

        -   type:       draco.synthesis.stream.SimulateSidereal
          # Generate a SiderealStream
            requires:   pm
            in:         imap
            out:        sstream
            params:
              # Save the output in the defined container format for SiderealStream
              save:   Yes
              output_root: '/data/zzhang/draco_out/sidereal_'

        -   type:       draco.synthesis.noise.GaussianNoise
            requires:   pm
            in:         sstream
            out:        sstream_wnoise
            params:
              recv_temp: 50
              ndays: 733
              save:   Yes
              output_root: '/data/zzhang/draco_out/sidereal_wnoise_'

        -   type:       draco.analysis.transform.MModeTransform
            in:         sstream_wnoise
            out:        mmodes
            params:
              save:   Yes
              output_root: '/data/zzhang/draco_out/mmode_'

        -   type:       draco.analysis.fgfilter.SVDModeProject
            requires:   pm
            in:         mmodes
            out:        svdmodes


        -   type:       draco.analysis.fgfilter.KLModeProject
            requires:   pm
            in:         svdmodes
            out:        klmodes
            params:
                klname:    dk_1thresh_fg_100thresh
                save:   Yes
                output_root: '/data/zzhang/draco_out/klmode_'
```

4. Run "maximum_likelihood.py" to acheive the maximum likelihood power spectrum using simulated data.
```
srun --mpi=pmi2 python maximum_likelihood.py
```





