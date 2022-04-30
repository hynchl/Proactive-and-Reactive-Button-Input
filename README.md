# Quantifying Proactive and Reactive Button Input

This repository contains analysis codes from [our project](https://dl.acm.org/doi/10.1145/3491102.3501913).


# Data
### Study 1 Dino (N=20) [Download](https://www.dropbox.com/sh/c2n7paqg7s4n0tw/AACho-njrUw7dn0IM0WkliAGa?dl=0)  
- `./S2**.mp4` : recorded screen. `**` is participant id.
- `./S2**.csv` : recorded input. `**` is participant id.  

### Study 1 Tabsonic (N=20) [Download](https://www.dropbox.com/sh/n2tz5e8mu2oocpr/AAAz5lk4Yj43FK3UCd-Q0OtWa?dl=0) 
- `./S2**_$%.mp4` : recorded screen. `**`, `$`, and `%` is the particiapnt id, song, and sound condition, respectively.
- `./S2**_$%.csv` : recorded input. `**`, `$`, and `%` is the particiapnt id, song, and sound condition, respectively.

### Study 2 Expanding Target Acquisition (N=12) [Download](https://www.dropbox.com/sh/lbmd5sq4kkn5ac2/AADtRqp7DZvwUb0W3hJuMbZta?dl=0)
- `./behavioral/S1**S.csv` : behavioral data logged in experiment app. `**` is participant id.
- `./behavioral/S1**S.json` : condition order presented in the experiments. `**` is participant id.
- `./S1**_$%.mp4` : recorded screen. `**`, `$`, and `%` is the particiapnt id, duration condition, and easing condition, respectively.
- `./S1**_$%.csv` : recorded input. `**`, `$`, and `%` is  the particiapnt id, duration condition, and easing condition, respectively.

If the dataset is not available, please email me (hyunchul.kim@kaist.ac.kr).



## Setting Environment

Conda virtual environment should be available in your computer.
Also, you need CUDA for installing cupy (CUDA 11.3)

```
conda create -n prb python=3.9 # you can replace 'prb' with a name you want.
conda activate prb
conda install cudatookit=11.3
pip install -r requirements.txt
```


## Logging
If you want to make your own dataset, please refer to [This](https://github.com/hynchl/obs-input-logger). Or, you can download our data above.


## Run Preprocessing
To analyze keyboard inputs, you should preprocess. If you want to analyze mouse inputs, you can skip this procedure.

```bash
# example
python key_preprocessing.py --task dino
python key_preprocessing.py --task tabsonic
```


## Run Extraction
We extract input-to-output intervals(IOI) from the recorded and input logs. Inputs are video in mp4 and button input logs in cvs collected by OBS and our plug-in script. Output is sequence of [x, y, ioi] in hdf5 format. 

```bash
# example
conda activate prb
python extract.py --path rawdata/dino/S201 --device keyboard --task dino --divider 32 --keys space --chunk 15000

python extract.py --path rawdata/tabsonic/S201_AO --device keyboard --task tabsonic --keys all --chunk 15000 
python extract.py --path rawdata/et/S101_32 --device mouse --task et --divider 32 --chunk 500 --relative True

```


## Run EM Fitting
This process fits the IOI data into our mixture model. 

```bash
# example
conda activate prb
python fit.py --task dino --pid S201 --key space
python fit.py --task tabsonic --pid S201 --sound O --key all
python fit.py --task et --pid S101 --easing 2 --duration 3
```

The output is the map of weight and parameters. For more detail, please refer to our notebooks for analysis.




