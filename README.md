**This is the repo for Rope3D Dataset. 
—— CVPR2022: Rope3D: The Roadside Perception Dataset for Autonomous Driving and Monocular 3D Object Detection Task).** [[Paper Link]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_Rope3D_The_Roadside_Perception_Dataset_for_Autonomous_Driving_and_Monocular_CVPR_2022_paper.pdf)


# Introduction
This repo is for releasing the Rope3D Dataset, which is the first high-diversity challenging **Ro**adside **Pe**rception 3D dataset- **Rope3D** from a novel view. The dataset consists of 50k images and over 1.5M 3D objects in various scenes, which are captured under different settings including various cameras with ambiguous mounting positions, camera specifications, viewpoints, and different environmental conditions.

The comparison between the roadside view and the conventional vehicle view can be see from the Figure below.

<img src="https://github.com/liyingying0113/rope3d-dataset-tools/blob/main/Examples/fig_different_view.png" width="600px">

Figure 1. The comparison of (a) frontal view and (b) roadside camera view with a pitch angle. The car view focuses more on the frontal area whereas the roadside camera observes the scene in a long-term and large-range manner. Vehicles can be easily occluded by closer objects in frontal view but the roadside view alleviates the risk. For example, for car-view (a), the white van is occluded by the black jeep whereas in roadside view (b) they are both visible, corresponding to the white and pink 3D boxes in (c). The triangle mark denotes the same LiDAR-mounted vehicle.


# Examples
![image](https://github.com/liyingying0113/rope3d-dataset-tools/blob/main/Examples/fig_examples_weather.png)


# Detailed Description and Download Website.
Please refer to :  [[Rope3D Dataset]](https://thudair.baai.ac.cn/rope)

Note that concurrently only regions within China can access to the dataset due to the policy constraint.

# Developing Tools
- See the `show_tools` folder for visualization the 3D bounding boxes.
- See the `eval_tools` folder for evaluating your results.


# Citation
If you find the dataset useful in your research, please cite using the following BibTex entry.  
@inproceedings{ye2022rope3d,  
  title={Rope3D: The Roadside Perception Dataset for Autonomous Driving and Monocular 3D Object Detection Task},  
  author={Ye, Xiaoqing and Shu, Mao and Li, Hanyu and Shi, Yifeng and Li, Yingying and Wang, Guangjie and Tan, Xiao and Ding, Errui},  
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},  
  pages={21341--21350},  
  year={2022}. 
}. 

