# Model Card Overview

3D-VLA is a generative world model that connects 3D perception, language, and action planning for Embodied AI. It uses 3D-based language models and embodied diffusion models to predict goal images and point clouds, enabling better reasoning and planning in 3D environments. Below, we introduce our models trained on diverse datasets.

| Model Name          | Link                                                                     | Task                                |
| ------------------- | ------------------------------------------------------------------------ | ----------------------------------- |
| Goal Image LDM      | [Huggingface](https://huggingface.co/anyezhy/3dvla-diffusion)            | RGB Goal Image Generation           |
| Goal RGB-D LDM      | [Huggingface](https://huggingface.co/anyezhy/3dvla-diffusion-depth)      | RGB-D Goal Image Generation         |
| Goal PC DM          | [Huggingface](https://huggingface.co/anyeZHY/3dvla-diffusion-pointcloud) | Point Cloud Goal Generation         |
| 3D-VLA LLM backbone | Coming Soon                                                              | 3D Vision-Language-Action Alignment |
| 3D-VLA              | Coming Soon                                                              |                                     |

# Model Details
## Goal Image LDM
**Model Description**: This model is a stable-diffusion-based generative model that generates RGB goal images from a given image and a text instruction. Trained on 192 V100s for 360000 steps for 3 weeks.

**Datasets**: RT-1 (Fractal data), Bridge, UCSD Picknplace, BC_Z, Epic Kitchen, CMU Playing With Food, Taco Play, Utaustin Mutex, Droid, Jaco Play, and Roboturk

## Goal RGB-D LDM

**Model Description**: This model is a stable-diffusion-based generative model that generates RGB + Depth goal images from RGB-D images and a text instruction. Trained on 96 V100s for 100000 steps for 2 weeks. We noticed that the model converges faster than the RGB model.

**Datasets**: RT-1 (Fractal data), Bridge, UCSD Picknplace, BC_Z, Epic Kitchen, CMU Playing With Food, Taco Play, Utaustin Mutex, Jaco Play, and Roboturk (without Droid)

## Goal PC DM

**Model Description**: This model is based on Point-E and generates point clouds from initial point clouds and text instructions. We modified the channels for the input and output layers to adapt to the point cloud generation task. The model was trained on 32 V100 GPUs for 200 epochs. During training, we used 2048 points, and we found that during inference, the model can generate point clouds larger than 2048 points, up to 8192 points.

**Datasets**: RLBench, including the following tasks: hang_frame_on_hanger, turn_tap, wipe_desk, push_buttons, put_money_in_safe, pick_up_cup, slide_block_to_target, close_door, lamp_on, water_plants, put_knife_on_chopping_board, turn_oven_on, take_frame_off_hanger, place_hanger_on_rack, scoop_with_spatula, open_drawer, reach_and_drag, close_drawer, sweep_to_dustpan, change_clock, take_umbrella_out_of_umbrella_stand, stack_wine, take_money_out_safe

**Data Preprocessing**: We used [Hiveformer](https://github.com/vlc-robot/hiveformer) to generate our data. For each task, we selected variation 0 for training and used the first text instructions from [`taskvar_instructions`](https://github.com/vlc-robot/hiveformer/blob/main/assets/taskvar_instructions.jsonl). Point clouds were preprocessed by removing the table background and normalizing them to fit within a unit sphere.


# Future Plans
We will release further models and details in the future. Stay tuned!

1. **3D-VLA LLM backbone**: 3D-LLM backbone that aligns 3D vision, language, and action.
2. **3D-VLA**: The full 3D-VLA model.