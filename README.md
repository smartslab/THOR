# Persistent Homology Meets Object Unity: Object Recognition in Clutter

Code repository for THOR presented in Persistent Homology Meets Object Unity: Object Recognition in Clutter.

## Dataset

The UW Indoor Scenes (UW-IS) Occluded Dataset proposed in the above paper for systematically evaluating object recognition methods under varying environmental conditions can be found [here](https://doi.org/10.6084/m9.figshare.20506506). This dataset, which is recorded using commodity hardware, consists of two different indoor environments, multiple lighting conditions, and multiple degrees of clutter.

## Requirements
* [Panda3D](https://www.panda3d.org/)
* Open3D
* persim 0.3.1
* scikit-learn
* Keras
* Platform: This code has been tested on Ubuntu 18.04 (except synthetic data generation using Panda3D, which is done on a computer running Windows 10).


## Usage
* ### Training:

1. Follow instructions from [here](https://docs.panda3d.org/1.10/python/introduction/installation-windows) to install Panda3D for synthetic data generation. Create a new folder `syndata` in the directory where Panda3D is installed and place `generateSyntheticData.py` from the training folder in this repository into the `syndata` folder. Create subfolders `models` and `data` inside the `syndata` folder. Within `models`, create subfolders for all objects and place respective object meshes and texture maps inside them. Obtain synthetic depth images from the object meshes using Panda3D using the following command. 

	```bash
	python generateSyntheticData.py --obj_name <obj_name> --h <h> --p <p> --r <r>
	```
	`<obj_name>` is the name of object for which data is to be generated, and the parameters `h`,`p`, and `r` are set to reorient the object mesh as required (details in the paper) before rendering. This command will create synthetic depth (and RGB) images for the object under a subfolder `<obj_name>` inside the `data` folder.

2. From within the THOR directory run the following to generate point clouds corresponding to all the generated depth images. 
	```bash
	python3 training/getPCDsFromSyntheticData.py --data_path <path_to_data_folder_from_step_one>
	```
3. From within the THOR directory run the following to perform view normalization on the generated point clouds. 
	```bash
	python3 training/saveAllViewNormalizedPCDs.py --data_path <path_to_data_folder_from_step_one>
	```
4. From within the THOR directory run the following to generate Persistence Images (PIs) for the TOPS descriptor of all the point clouds.
	```bash
	python3 training/computePIsFromViewNormalizedPCDs.py --data_path <path_to_data_folder_from_step_one>
	```
	A subfolder named `libpis` containing all the PIs will be generated inside the  `training` folder .

5. Run the following to train an SVM library using the TOPS descriptors obtained from the computed PIs. (Add the path to the data folder from step 1 in `trainSVMLibrary.sh` as indicated).

	```bash
	cd training
	sh trainSVMLibrary.sh
	```
	Alternatively, to train an MLP library run the following.  (Add the path to the data folder from step 1 in `trainMLPLibrary.sh` as indicated).
	```bash
	cd training
	sh trainMLPLibrary.sh
	```
	 A folder `librarymodels` will be created inside the `training` directory and trained models will be stored in it.

* ### Testing on the UW-IS Occluded Dataset:
1. Download the UWISOccludedDataset.zip from [here](https://doi.org/10.6084/m9.figshare.20506506) and unzip it. Place `reogranizeUWISOccluded.sh` inside the `UWISOccludedDataset` folder and run the following from within that folder.

	```bash
	sh reorganizeUWISOccluded.sh
	```
2. Run the following to test THOR
	* Using an SVM library:
		```bash
		cd testing
		sh testUWISOccludedSVMLibrary.sh
		```
		Note that in the `testUWISOccludedSVMLibrary.sh` script `<environment_name>` must be replaced with one of  `warehouse`,`lounge`, or `both` as desired. Similarly `<category_name>` can be `kitchen`, `food`, `tools`  or `all`; `<separation>` can be `level1`, `level2`, `level3`  or `alllevels`; `<light>` can be `1`,`2`, or `both`.  Also provide the path to the UW-IS Occluded dataset folder from the previous step, and the path to the folder containing saved SVM models.
		 
		A subfolder named `predictions` is created in the `testing` folder and predictions for every video will be saved as a `.txt` file. Corresponding ground truth will be saved in a newly created `groundtruth` subfolder.

	* Using an MLP library:
		```bash
		cd testing
		sh testUWISOccludedMLPLibrary.sh
		```
		Note that `<environment_name>`,  `<category_name>`, `<separation>`, and `<light>` are to be replaced as described above. The path to the dataset folder and the folder containing trained MLP models must also be provided. As in the SVM case, a subfolder named `predictions` is created in the `testing` folder and predictions for every video will be saved as a `.txt` file. Corresponding ground truth will be saved in a newly created `groundtruth` subfolder.


