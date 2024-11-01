# AutoSAM
AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder

## Overview

This work improves the Segment Anything Model (SAM) for medical image segmentation by replacing its conditioning mechanism with an image-based encoder. Without further fine-tuning SAM, this modification achieves state-of-the-art results on medical images and video benchmarks. 

## Paper

The paper associated with this repository can be found [here](https://arxiv.org/pdf/2306.06370.pdf).

## Datasets

We used the following datasets in our experiments:

[monu](https://drive.google.com/drive/folders/1bzyHsDWhjhiwzpx_zJ5dpMG3-5F-nhT4?usp=drive_link)
[glas](https://drive.google.com/drive/folders/1z9xBesNhvuM08yUOpOWcUy7OnBGHenFv?usp=drive_link)
[polyp](https://drive.google.com/drive/folders/1S11HsauwKO206CPzrGBnTid-nbQMhbZz?usp=drive_link)

## SAM checkopints

[sam base](https://drive.google.com/file/d/1ZwKc-7Q8ZaHfbGVKvvkz_LPBemxHyVpf/view?usp=drive_link)
[sam large](https://drive.google.com/file/d/16AhGjaVXrlheeXte8rvS2g2ZstWye3Xx/view?usp=drive_link)
[sam huge](https://drive.google.com/file/d/1tFYGukHxUCbCG3wPtuydO-lYakgpSYDd/view?usp=drive_link)

## Usage

To use AutoSAM, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/AutoSAM.git
   cd AutoSAM/

2. conda:

   ```bash
   conda env update -n autosam --file environment.yml

3. training:
   Downloading the prompt encoder checkpoint (trained on ImageNet) will fail so for this talk to me
   ```bash
   Cup:
   python train.py --task=retina --epoches=5 --order=68 -t cup

   Disk:
   python train.py --task=retina --epoches=5 --order=68 -t cup

4. Annotation (`python auto_annotator.py -i [INPUT_IMAGE_PATH] -c [PROMPT_ENCODER_CHECKPOINT] -o [IMAGE_OUT_PATH]`) E.g.:
   ```bash
   python auto_annotator.py -i /dhc/dsets/REFUGE/REFUGE/Test-400/0403/0403.jpg -c /dhc/home/tobias.fiedler/AutoSAM/results/cup/gpu2/net_best.pth -o cup.png