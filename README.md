# det3d-readme
This is a more detailed instruction of how to get [det3d](https://github.com/GatikAI/det3d/tree/98c86de7dd88f569e430368761b0c5eca6568672) run.

Suppose the data is at a shared directory on the server: */real_data_path*

#### 1. build an image with your own tag (e.g. det3d_dev_ting)
```
docker build . -t det3d_dev_ting
```
#### 2. make a output directory to store the checkpoint and logs, here we use /real_output_path

#### 3. run an container named det3d_ting(or whatever you like)
Environment variables can be used to specify data paths for server vs. Cloud (not sure about how to use this yet)
```
docker run -it --gpus all -v /real_data_path:/data -v /real_output_path:/output -v $(pwd):/src/det3d -e "BASE_DATA_DIR=/data" -e "BASE_INFO_DIR=/data/infos_v2" --net=host --name det3d_ting --rm det3d_dev_ting bash
```

#### 4. preprocess data, here we use the dataset specified by config file: data_loader_deepen.yaml
```
# Use tools/cfgs/data_configs/data_loader_deepen.yaml
python tools/preprocess_data.py --cfg_file tools/cfgs/data_configs/data_loader_deepen.yaml
python pcdet/datasets/processors/object_db_sampler/prepare_object_db.py --cfg_file tools/cfgs/data_configs/data_loader_deepen.yaml
```
the first command convert the format of the meta info of the dataset and save it into .pkl file; the second prepare the sample_data_set according to category and during training random samples will be drawn from sample_data_set to augment the training data.


#### 5. Visualize the dataset

##### a) visualize the raw data
The visualizer directly use dataset and its groundtruth annotation to visualize it. 
see instructions of [viz_3d](https://github.com/GatikAI/det3d/tree/master/viz_3d)

After running the visualizer(before SSH port forwarding), an IP address and port will show in the terminal telling you where the visualizer is running.  If you are in office, you can directly access it from your browser.  Otherwize you need to do SSH port forwarding.  Since we access the server via **Cloudflare**, we need to config our terminal to go through the same connection, see [Using the Terminal](https://gatik.atlassian.net/wiki/spaces/IDD/pages/1604321299/SSH+-+Cloudflare+for+Teams#Using-the-Terminal).  Notice that *'ssh server01.gatik.ai'* will access the server with the same user name you use with local computer, for a different user name, you can either access the server with *'ssh username.on.server@server01.gatik.ai'* or add *'user username.on.server'* in *~/.ssh/config*.  Just for your info : it's not possible to configure a default password in an ssh config file.

Once the visualizer is started, you need to input the dataset config file (there is a default file there though), click **load** once, and then click **>** (next frame) to see the data, play with the size and color to get a better view, press&hold **shift** and right click&hold&drag the mouse to rotate the view.

##### b) visualize the statistics of multiple dataset 
run the */src/det3d/notebooks/dataset_statistics.ipynb* to see the statistics


#### 6. Run training
The default working directory is */src/det3d/tools*.
Suppose we are using the config of *deepen_cbgs_pp_multihead_extended_range_all_classes_T4x4.yaml*, running command from dirctory of *src/det3d/tools*, the training command would be :
```
root@tor-testbench01:/src/det3d/tools# python train.py --cfg_file cfgs/deepen_models/deepen_cbgs_pp_multihead_extended_range_all_classes_T4x4.yaml
```
It is intended for 4 x T4 GPUs, so you may have to adjust the batch_size and number of epochs accordingly.
Optionally, if you want to use a specified GPU (e.g GPU 1): 
```
root@tor-testbench01:/src/det3d/tools# CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/deepen_models/deepen_cbgs_pp_multihead_extended_range_all_classes_T4x4.yaml
```
Once that is working, try multi-gpu. the first number argument for the .sh file is the number of GPUs
```
./scripts/dist_train.sh 2 .....
```

#### 7. Launch tensorbvoard to visualize training curves.
the default port is 6007.
```
tensorboard --host 0.0.0.0 --logdir /output
```

You'll have to use the same port forwarding trick as with the visualization tool:
```
ting@ting-ThinkPad-X1-Extreme-Gen-3:~$ ssh -vL 6007:127.0.0.1:6007 ting.sun@tor-testbench01-s.gatik.ai
```

#### 8. Eval example
```
python test.py --cfg_file cfgs/deepen_models/nuscenes_pretrain/cbgs_pp_multihead_deepen_finetune.yaml --ckpt /output/deepen_models/nuscenes_pretrain/cbgs_pp_multihead_deepen_finetune/default/ckpt/checkpoint_epoch_20.pth
```

#### 9. GPU memory release
If the stopped process does not release the GPU memory automatically, you have to manually kill the process. in a terminal conneccted to the server but not attach to the container run *'nvidia-smi'* to see the process that occupies the GPU memory, use *'sudo kill pid'* to kill it. (run *nvidia-smi* inside the container will not show the process id)

#### 10. visual studio code remote debugging
you can install **visual studio code** in your local computer and debug the code on the server using the environment of a docker container running on server, detailed instructions can be found at [Visual Studio Code Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
