# Quantifying-Proactive-and-Reactive-Button-Input

This repository contains analysis codes from [our project](...).


# Data
### Study 1 Dino (N=20) [Download](...)  
- `./S2**.mp4` : recorded screen. ** is participant id.
- `./S2**.csv` : recorded input. ** is participant id.  

### Study 1 Tabsonic (N=20) [Download](...) 
- './S2**_$%.mp4' : recorded screen. **, $, and % is the particiapnt id, song, and sound condition, respectively.
- './S2**_$%.csv' : recorded input. **, $, and % is the particiapnt id, song, and sound condition, respectively.

### Study 2 Expanding Target Acquisition (N=12) [Download](...)
- './ET_BEHAVIORAL/S1**S.csv' : behavioral data logged in experiment app. ** is participant id.
- './ET_BEHAVIROAL/S1**S.json' : condition order presented in the experiments. ** is participant id.
- './S1**_$%.mp4' : recorded screen. **, $, and % is the particiapnt id, duration condition, and easing condition, respectively.
- './S1**_$%.csv' : recorded input. **, $, and % is  the particiapnt id, duration condition, and easing condition, respectively.

If dataset is not available, please email us (byungjoo.lee@yonsei.ac.kr or hyunchul.kim@kaist.ac.kr).

# Pipeline

<<image>>

## Setting Environment

Conda virtual environment should be available in your computer.

```
conda create -n prb python 3.7 # you can replace 'prb' with a name you want.
conda activate prb
pip install -r requirements.txt
```

## Logging
If you want to make your own dataset, please refer to [This](https://github.com/hynchl/obs-input-logger). Or, you can download our data.

## Run Preprocessing
We extract input-to-output intervals(IOI) from the recorded and input logs. Inputs are video in mp4 format and button input logs in cvs format collected with OBS and our plug-in script. Outpus is sequence of [x, y, ioi] in hdf5 format. 
```
conda activate prb
python preprocess.py --path {file path} --input {'mouse' or 'keyboard'}
```

## Run EM Fitting
This process fits the IOI data into our mixture model. The output is the map of [weight of Proactiveness, weight of Reactiveness, weight of Irrelevance, mu_P, sigma_P, ..., ..., ..., ...].

```
conda activate prb
python fit.py --path --path {file path}
```

# Citation
```tex
{}
```




