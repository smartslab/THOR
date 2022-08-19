# Persistent Homology Meets Object Unity: Object Recognition in Clutter

Code repository for the methods presented in Persistent Homology Meets Object Unity: Object Recognition in Clutter.

## Dataset

The UW Indoor Scenes (UW-IS) Occluded Dataset proposed in the above paper for systematically evaluating object recognition methods under varying environmental conditions can be found [here](https://doi.org/10.6084/m9.figshare.20506506). This dataset, which is recorded using commodity hardware, consists of two different indoor environments, multiple lighting conditions, and multiple degrees of clutter.

## Requirements
* [Panda3D](https://www.panda3d.org/)
* Open3D
* persim 0.3.1
* scikit-learn
* Keras

## Usage
* ### Training:

1. Use `generateSyntheticData.py` to obtain synthetic depth images for objects using Panda3D
2. Generate point clouds from synthetic depth images using `getPCDsFromSyntheticData.py`, perform view normalization using `saveAllViewNormalizedPCDs.py`, and compute PIs for the TOPS descriptor using `computePIsFromViewNormalizedPCDs.py`
3. Train an SVM (using `trainSVMLibrary.py`) or MLP (using `trainMLPLibrary.py`) library

* ### Testing:
1. Use `testUWISOccluded.py` to test on the UW-IS Occluded Dataset using the generated library
