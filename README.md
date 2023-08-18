# Quick Start

This code repository includes the base run code for AMSP-UOD, see folder `./weights` for the weights file, and see folder `./result` for the PR curve graph. There is some of the urpc test data in folder `./urpc`, and you can get the recognition results using `./det.sh`

## 1. Deploy Conda environment
```Command Line
conda create -n AMSP_UOD python==3.10
```

## 2. Install package dependencies
```Command Line
pip install -r requirements.txt
```

## 3. Train Model (Optional, requires Datasets and Cuda)
```Command Line
conda activate AMSP_UOD
./train.sh 0
```

## 4. Test Model (Optional, requires Datasets and Cuda)
```Command Line
conda activate AMSP_UOD
./val.sh
```

## 5. Dectet
```Command Line
conda activate AMSP_UOD
./det.sh
```

# Showcase

## URPC
![img1](./result/Traditional-NMS/URPC-Ours_AMSP_UOD.png)
## RUOD
![img2](./result/Traditional-NMS/RUOD-Ours_AMSP_UOD.png)