# Model Card Overview

3D-VLA is a generative world model that connects 3D perception, language, and action planning for Embodied AI. It uses 3D-based language models and embodied diffusion models to predict goal images and point clouds, enabling better reasoning and planning in 3D environments. Below, we introduce our models trained on diverse datasets.

| Model Name          | Link                                                                | Task                                |
| ------------------- | ------------------------------------------------------------------- | ----------------------------------- |
| Goal Image LDM      | [Huggingface](https://huggingface.co/anyezhy/3dvla-diffusion)       | RGB Goal Image Generation           |
| Goal RGB-D LDM      | [Huggingface](https://huggingface.co/anyezhy/3dvla-diffusion-depth) | RGB-D Goal Image Generation         |
| Goal PC DM          | Coming Soon                                                         | Point Cloud Goal Generation         |
| 3D-VLA LLM backbone | Coming Soon                                                         | 3D Vision-Language-Action Alignment |
| 3D-VLA              | Coming Soon                                                         |                                     |

# Model Details
## Goal Image LDM
**Model Description**: This model is a stable-diffusion-based generative model that generates RGB goal images from a given image and a text instruction. Trained on 192 V100s for 360000 steps for 3 weeks.

**Datasets**: RT-1 (Fractal data), Bridge, UCSD Picknplace, BC_Z, Epic Kitchen, CMU Playing With Food, Taco Play, Utaustin Mutex, Droid, Jaco Play, and Roboturk

## Goal RGB-D LDM

**Model Description**: This model is a stable-diffusion-based generative model that generates RGB + Depth goal images from RGB-D images and a text instruction. Trained on 96 V100s for 100000 steps for 2 weeks. We noticed that the model converges faster than the RGB model.

**Datasets**: RT-1 (Fractal data), Bridge, UCSD Picknplace, BC_Z, Epic Kitchen, CMU Playing With Food, Taco Play, Utaustin Mutex, Jaco Play, and Roboturk (without Droid)

# Future Plans
We will release further models and details in the future. Stay tuned!

1. **Goal PC DM**: A Point-E-based model that generates point clouds from initial point clouds and text instructions. Trained on RLBench dataset.
2. **3D-VLA LLM backbone**: 3D-LLM backbone that aligns 3D vision, language, and action.
3. **3D-VLA**: The full 3D-VLA model.