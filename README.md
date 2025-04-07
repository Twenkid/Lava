# Lava
Lava-nc Neuromorphic library - Study, Exercises, experiments


5.4.2025, Next: see https://lava-nc.org/lava-lib-dl/bootstrap/notebooks/mnist/train.html  |  
https://r-gaurav.github.io/2024/04/13/Lava-Tutorial-MNIST-Training-on-GPU-and-Evaluation-on-Loihi2.html

7.4.2025: Run on Win10, Conda:

![image](https://github.com/user-attachments/assets/fdd7585d-6466-41b9-8f31-38975d2b3afe)

Starting attempts to convert to accelerate CUDA version, but the documentation so far is incomplete.

conda activate clip
cd c:/py/2025/lava
python mt.py

Switching to Loihi2SimCfg, but not installed torch with CUDA support.
So far I didn't find info about what specific version of Torch with CUDA is supported and there are issues with the Conda.
Pytorch 2.6 with CUDA 12.6 installed, but the modified code seems not to utilize it.
Changing select_tag to "floating_pt" or removing it leads to ~ equally distributed results, noise, so it should be fixed_pt.
My GPU Geforce 750 Ti doesn't support fixed_pt (10xx series support 8 bit INT ), but it is not utilized with float either.

More research is needed. 

Check a reference: https://lava-nc.org/lava/lava.magma.core.html#lava.magma.core.run_configs.AbstractLoihiRunCfg

etc.

mnist_clf.run(
            condition=RunSteps(num_steps=num_steps_per_image),
            run_cfg=Loihi2SimCfg(select_sub_proc_model=True)) #,select_tag='fixed_pt'))

```
gt_label= [3]
3
1 (1,) 4 [43. 43. 43. 43. 45. 43. 43. 43. 45. 45.]
[3] 3
self.current_img_id=27
Current image: 29
run_post_mgmt: self.ground_truth_label = 2
 self.ground_truth_label=2
gt_label= [2]
2
1 (1,) 4 [43. 43. 43. 43. 45. 43. 43. 43. 45. 45.]
[2] 2
self.current_img_id=28
Current image: 30
run_post_mgmt: self.ground_truth_label = 7
 self.ground_truth_label=7
gt_label= [7]
7
1 (1,) 4 [43. 43. 43. 43. 45. 43. 43. 43. 45. 45.]
[7] 7
self.current_img_id=29

Ground truth: [5 0 4 1 9 2 1 3 1 4 3 5 3 6 1 7 2 8 6 9 4 0 9 1 1 2 4 3 2 7]
Predictions : [4 4 4 4 4 4 4 4 9 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 9 4 4 4 4]
Accuracy    : 13.333333333333334
```







