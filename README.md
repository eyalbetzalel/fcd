# FCD
 Fréchet Clip Distance Implementation for PyTorch. 
 
 The FID is a measure of similarity between two sets of images. In studies on the quality of samples of GANs, this measure has been found to correlate well with human judgments of visual quality. The Fréchet distance is calculated between two Gaussians fitted to feature representations of the Inception network.  
 We showed in our article that CLIP feature more suitable for this task under different datasets. 
 
 This code is based on *mseitzer* FID implementation that can be found [here](https://github.com/mseitzer/pytorch-fid)

## Installation

```pip install -r requirements.txt```

## Usage

```python fcd.py --path_source /path/to/source/ --path_test /path/to/test/```

folders containing PNG format images.  

## Citing

If you find our work or this code to be useful in your own research, please consider citing the following paper:

```@misc{https://doi.org/10.48550/arxiv.2206.10935,
  doi = {10.48550/ARXIV.2206.10935},
  
  url = {https://arxiv.org/abs/2206.10935},
  
  author = {Betzalel, Eyal and Penso, Coby and Navon, Aviv and Fetaya, Ethan},
  
  title = {A Study on the Evaluation of Generative Models},
  
  publisher = {arXiv},
  
  year = {2022}
  
}
