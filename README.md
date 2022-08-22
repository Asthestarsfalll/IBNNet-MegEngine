# IBNNet-MegEngine

The MegEngine implementation of IBNNet(Two at Once: Enhancing Learning and Generalization Capacities via *IBN*-*Net*)

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't want to compare the ouput error between the MegEngine implementation and PyTorch one, just ignore requirements.txt and install MegEngine from the command line:

```bash
python3 -m pip install --upgrade pip 
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

### Convert Weights

Convert trained weights from torch to megengine, the converted weights will be save in ./pretained/ .

```bash
python convert_weights.py -m  densenet121_ibn_a
```

If the download speed is too slow, you may download them manually.

### Compare

Use `python compare.py` .

### Load From Hub

Import from megengine.hub:

Way 1:

```python
from megengine import hub

# load pretrained model
pretrained_model = modelhub.resnet50_ibn_a(pretrained=True)
```

Way 2:

```python
from  megengine import hub

# load pretrained model 
model_name = 'resnet50_ibn_a'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/IBNNet-MegEngine:main', entry=model_name, git_host='github.com', pretrained=True)
```

For those models which do not have pretrained model online, you need to convert weights mannaly,  and load the model without pretrained weights like this:

```python
model = modelhub.resnet50_ibn_a()
# or
model_name = 'resnet50_ibn_a'
model = hub.load(
    repo_info='asthestarsfalll/IBNNet-MegEngine:main', entry=model_name, git_host='github.com')
 
model.load_state_dict(mge.load("path/to/weight"))
```

## Reference

[The official pytorch implementation](https://github.com/XingangPan/IBN-Net)