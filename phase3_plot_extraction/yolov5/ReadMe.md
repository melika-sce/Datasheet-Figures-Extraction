# Diagram Detector

Detect diagrams in pdf files and convert them into digital diagrams

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

1. Navigate to the main folder containing all the code:

```bash
cd code/DiagramDetection
```
2. create conda env
```bash
conda create -n det python=3.8 -y
conda activate det
```
3. install mmdet and dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

%cd DiagramDetection/mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"

pip install img2pdf --no-cache-dir
pip install pdf2jpg
pip install easyocr
pip install mmcv==1.3.17
pip install mmcv-full
```


## Usage

### phase 1 - detect plots in pdfs

#### Train:
In order to train the diagrem detector:
- change the data path in `data/Diagram.yaml` to your own data path
- run the bellow command

```bash
python train.py --img 1280 --batch 16 --epochs 100 --data data/Diagram.yaml --weights model/Diagram-detector-best.pt
```

#### Inference
In order to use the detector in inference mode:
- put your pdfs in `sample_pdfs` path
- run the command below (note that all variables have default values, but you can specify them too)

```bash
!python diagram_det.py --input_folder sample_pdfs --model_weight_address model/Diagram-detector-best.pt --conf_threshold 0.85 --result_path runs/detect/diagrams/
```

### phase 2 - detect plot-area, x-axis and y-axis

#### Train:
In order to train the diagrem detector:
- change the data path in `data/labelChart.yaml` to your own data path
- run the bellow command

```bash
python train.py --img 1280 --batch 16 --epochs 100 --data data/labelChart.yaml --weights model/labelChart-detector-best.pt
```

#### Inference
- run the command below (note that all variables have default values, but you can specify them too)

```bash
!python diagram_label_det.py --input_folder runs/detect/diagrams --model_weight_address model/labelChart-detector-best.pt --conf_threshold 0.50 --result_path runs/detect/diagram_labels/
```


### phase 3 - detect lines and plot data (legends, axis numbers and labels)

#### Train:
In order to train the diagrem detector:
- change the data path in `data/legend.yaml` to your own data path
- run the bellow command

```bash
python train.py --img 1280 --batch 16 --epochs 100 --data data/legend.yaml --weights model/legend-detector-best.pt
```

#### Inference
- run the command below (note that all variables have default values, but you can specify them too)

```bash
!python label_chart.py --input_folder runs/detect/diagram_labels --model_weight_address model/legend-detector-best.pt --conf_threshold 0.50 
```

### phase 4 - plot final results
- run the command below (note that all variables have default values, but you can specify them too)

```bash
!python make_plot.py --input_path runs/detect/diagram_labels --output runs/detect/plots
```

## easyocr error handling
in case getting this error: `    img = cv2.resize(img,(model_height,int(model_height*ratio)), interpolation=Image.ANTIALIAS)
AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'`

cahnge this line to: 
img = cv2.resize(img,(model_height,int(model_height*ratio)), interpolation=cv2.INTER_LANCZOS4)
