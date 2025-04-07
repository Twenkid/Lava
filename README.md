# Lava
Lava-nc Neuromorphic library - Study, Exercises, experiments


5.4.2025, Next: see https://lava-nc.org/lava-lib-dl/bootstrap/notebooks/mnist/train.html  |  
https://r-gaurav.github.io/2024/04/13/Lava-Tutorial-MNIST-Training-on-GPU-and-Evaluation-on-Loihi2.html

7.4.2025: Run on Win10, Conda:

![image](https://github.com/user-attachments/assets/fdd7585d-6466-41b9-8f31-38975d2b3afe)

Starting attempts to convert to accelerated CUDA version.

conda activate clip
cd c:/py/2025/lava
python mt.py

Switching to Loihi2SimCfg, but not installed torch with CUDA support.
So far I didn't find info about what specific version of Torch with CUDA is supported and there are issues with the Conda.
Pytorch 2.6 with CUDA 12.6 installed, but the modified code seems not to utilize it.
Changing select_tag to "floating_pt" or removing it leads to ~ equally distributed results, noise, so it should be fixed_pt.
My GPU Geforce 750 Ti doesn't support fixed_pt (10xx series support 8 bit INT ), but it is not utilized with float either.

More research is needed - continue with the above mentioned resources.

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

**7.4.2025 - 8.4.2025 1:25**

* I found how to run training with GPU acceleration, however all possible hacks didn't help to make it roll on 750 Ti - minimum 10xx series GPU (Pascal, SM 6.1).
It didn't work with CUDA 12.6, 11.2. Besides they are newer than what nvcc supports for 750 Ti (sm_50 (Maxwell)).
CUDA 10.2 - installed, adjusted the command line to use C++14, but the code requires C++17.

The run with CPU backend (Torch 2.3.1+cpu) was successful, but not waited.

Epoch: 1, Stats: Train loss =    24.39363                          accuracy = 0.84487  | Test  loss =    20.60499                          accuracy = 0.88370

python train_eval_snn.py --n_tsteps=20 --epochs=20 --backend=GPU
-->
python train_eval_snn.py --n_tsteps=20 --epochs=20 --backend=CPU
...


```
(clip) c:\PY\2025\mnist-on-loihi>python train_eval_snn.py --n_tsteps=20 --epochs=20 --backend=GPU
WARNING: Prophesee Dataset Toolbox could not be found!
         Only Prophesee DVS demo will not run properly.
         Please install it from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
********************************************************************************
Training and Evaluating SlayerDenseSNN on GPU ... AND,
Evaluating LavaDenseSNN on Loihi-2 Simulation Hardware on CPU.
********************************************************************************
WARNING: Prophesee Dataset Toolbox could not be found!
         Only Prophesee DVS demo will not run properly.
         Please install it from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
WARNING: Prophesee Dataset Toolbox could not be found!
         Only Prophesee DVS demo will not run properly.
         Please install it from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
WARNING: Prophesee Dataset Toolbox could not be found!
         Only Prophesee DVS demo will not run properly.
         Please install it from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
WARNING: Prophesee Dataset Toolbox could not be found!
         Only Prophesee DVS demo will not run properly.
         Please install it from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\utils\cpp_extension.py:2059: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
Traceback (most recent call last):
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\utils\cpp_extension.py", line 2209, in _run_ninja_build
    subprocess.run(
  File "C:\Users\toshb\.conda\envs\clip\lib\subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "c:\PY\2025\mnist-on-loihi\train_eval_snn.py", line 74, in <module>
    tes.train_eval_snn()
  File "c:\PY\2025\mnist-on-loihi\train_eval_snn.py", line 37, in train_eval_snn
    output = assistant.train(inp, lbl)
  File "C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\utils\assistant.py", line 121, in train
    output = self.net(input)
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "c:\PY\2025\mnist-on-loihi\net_utils\snns.py", line 52, in forward
    spikes = block(spikes)
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\block\base.py", line 547, in forward
    x = self.neuron(z)
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\neuron\cuba.py", line 439, in forward
    _, voltage = self.dynamics(input)
  File "C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\neuron\cuba.py", line 356, in dynamics
    current = leaky_integrator.dynamics(
  File "C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\neuron\dynamics\leaky_integrator.py", line 95, in dynamics
    output = Accelerated.leaky_integrator.dynamics(
  File "C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\utils\utils.py", line 22, in __get__
    return staticmethod(self.fget).__get__(None, owner)()
  File "C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\neuron\dynamics\leaky_integrator.py", line 40, in leaky_integrator
    Accelerated.module = load(
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\utils\cpp_extension.py", line 1380, in load
    return _jit_compile(
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\utils\cpp_extension.py", line 1798, in _jit_compile
    _write_ninja_file_and_build_library(
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\utils\cpp_extension.py", line 1926, in _write_ninja_file_and_build_library
    _run_ninja_build(
  File "C:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\utils\cpp_extension.py", line 2225, in _run_ninja_build
    raise RuntimeError(message) from e
RuntimeError: Error building extension 'dynamics': [1/2] C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc --generate-dependencies-with-compile --dependency-output leaky_integrator.cuda.o.d -Xcudafe --diag_suppress=dll_interface_conflict_dllexport_assumed -Xcudafe --diag_suppress=dll_interface_conflict_none_assumed -Xcudafe --diag_suppress=field_without_dll_interface -Xcudafe --diag_suppress=base_class_has_different_dll_interface -Xcompiler /EHsc -Xcompiler /wd4068 -Xcompiler /wd4067 -Xcompiler /wd4624 -Xcompiler /wd4190 -Xcompiler /wd4018 -Xcompiler /wd4275 -Xcompiler /wd4267 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4819 -Xcompiler /MD -DTORCH_EXTENSION_NAME=dynamics -DTORCH_API_INCLUDE_EXTENSION_H -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\torch\csrc\api\include -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\TH -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\THC "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include" -IC:\Users\toshb\.conda\envs\clip\Include -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_50,code=compute_50 -gencode=arch=compute_50,code=sm_50 -std=c++17 -c C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\neuron\dynamics\leaky_integrator.cu -o leaky_integrator.cuda.o
FAILED: leaky_integrator.cuda.o
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc --generate-dependencies-with-compile --dependency-output leaky_integrator.cuda.o.d -Xcudafe --diag_suppress=dll_interface_conflict_dllexport_assumed -Xcudafe --diag_suppress=dll_interface_conflict_none_assumed -Xcudafe --diag_suppress=field_without_dll_interface -Xcudafe --diag_suppress=base_class_has_different_dll_interface -Xcompiler /EHsc -Xcompiler /wd4068 -Xcompiler /wd4067 -Xcompiler /wd4624 -Xcompiler /wd4190 -Xcompiler /wd4018 -Xcompiler /wd4275 -Xcompiler /wd4267 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4819 -Xcompiler /MD -DTORCH_EXTENSION_NAME=dynamics -DTORCH_API_INCLUDE_EXTENSION_H -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\torch\csrc\api\include -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\TH -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\THC "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include" -IC:\Users\toshb\.conda\envs\clip\Include -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_50,code=compute_50 -gencode=arch=compute_50,code=sm_50 -std=c++17 -c C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\neuron\dynamics\leaky_integrator.cu -o leaky_integrator.cuda.o
nvcc fatal   : Value 'c++17' is not defined for option 'std'
ninja: build stopped: subcommand failed.

(clip) c:\PY\2025\mnist-on-loihi>C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc --generate-dependencies-with-compile --dependency-output leaky_integrator.cuda.o.d -Xcudafe --diag_suppress=dll_interface_conflict_dllexport_assumed -Xcudafe --diag_suppress=dll_interface_conflict_none_assumed -Xcudafe --diag_suppress=field_without_dll_interface -Xcudafe --diag_suppress=base_class_has_different_dll_interface -Xcompiler /EHsc -Xcompiler /wd4068 -Xcompiler /wd4067 -Xcompiler /wd4624 -Xcompiler /wd4190 -Xcompiler /wd4018 -Xcompiler /wd4275 -Xcompiler /wd4267 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4819 -Xcompiler /MD -DTORCH_EXTENSION_NAME=dynamics -DTORCH_API_INCLUDE_EXTENSION_H -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\torch\csrc\api\include -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\TH -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\THC "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include" -IC:\Users\toshb\.conda\envs\clip\Include -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_50,code=compute_50 -gencode=arch=compute_50,code=sm_50 -std=c++14 -c C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\neuron\dynamics\leaky_integrator.cu -o z:/leaky_integrator.cuda.o
'C:\Program' is not recognized as an internal or external command,
operable program or batch file.

(clip) c:\PY\2025\mnist-on-loihi>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc" --generate-dependencies-with-compile --dependency-output leaky_integrator.cuda.o.d -Xcudafe --diag_suppress=dll_interface_conflict_dllexport_assumed -Xcudafe --diag_suppress=dll_interface_conflict_none_assumed -Xcudafe --diag_suppress=field_without_dll_interface -Xcudafe --diag_suppress=base_class_has_different_dll_interface -Xcompiler /EHsc -Xcompiler /wd4068 -Xcompiler /wd4067 -Xcompiler /wd4624 -Xcompiler /wd4190 -Xcompiler /wd4018 -Xcompiler /wd4275 -Xcompiler /wd4267 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4819 -Xcompiler /MD -DTORCH_EXTENSION_NAME=dynamics -DTORCH_API_INCLUDE_EXTENSION_H -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\torch\csrc\api\include -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\TH -IC:\Users\toshb\.conda\envs\clip\lib\site-packages\torch\include\THC "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include" -IC:\Users\toshb\.conda\envs\clip\Include -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_50,code=compute_50 -gencode=arch=compute_50,code=sm_50 -std=c++14 -c C:\Users\toshb\AppData\Roaming\Python\Python310\site-packages\lava\lib\dl\slayer\neuron\dynamics\leaky_integrator.cu -o z:/leaky_integrator.cuda.o
nvcc warning : The -std=c++14 flag is not supported with the configured host compiler. Flag will be ignored.
leaky_integrator.cu
C:/Users/toshb/.conda/envs/clip/lib/site-packages/torch/include\c10/util/C++17.h(24): fatal error C1189: #error:  You need C++17 to compile PyTorch
```







