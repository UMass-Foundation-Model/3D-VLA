<br/>
<p align="center">
  <h1 align="center"><a style="color:#61a5c2;">3D</a>-<a style="color:#94D2BD;">V</a><a style="color:#EE9B00;">L</a><a style="color:#CA6502;">A</a>: A 3D Vision-Language-Action Generative World Model</h1>
  <p align="center">
    ICML 2024
  </p>
  <p align="center">
    <a href="https://haoyuzhen.com">Haoyu Zhen</a>,
    <a href="">Xiaowen Qiu</a>,
    <a href="https://peihaochen.github.io">Peihao Chen</a>,
    <a href="https://github.com/Yang-Chincheng">Jincheng Yang</a>,
    <a href="https://cakeyan.github.io">Xin Yan</a>,
    <a href="https://yilundu.github.io">Yilun Du</a>,
    <a href="https://evelinehong.github.io">Yining Hong</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2403.09631">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://vis-www.cs.umass.edu/3dvla' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Tabel of Contents</summary>
  <ol>
    <li>
      <a href="#method">Method</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#embodied-diffusion-models">Embodied Diffusion Models</a>
      <ul>
        <li><a href="#goal-image-generation">Goal Image Generation</a></li>
      </ul>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>

## Method

3D-VLA is a framework that connects vision-language-action (VLA) models to the 3D physical world. Unlike traditional 2D models, 3D-VLA integrates 3D perception, reasoning, and action through a generative world model, similar to human cognitive processes. It is built on the [3D-LLM](https://vis-www.cs.umass.edu/3dllm/) and uses interaction tokens to engage with the environment. Embodied diffusion models are trained and aligned with the LLM to predict goal images and point clouds.

<p align="center">
    <img src="docs/method.png" alt="Logo" width="80%">
</p>

## Installation

```bash
conda create -n 3dvla python=3.9
conda activate 3dvla
pip install -r requirements.txt
```

## Embodied Diffusion Models

### Goal Image Generation
- Train the goal image latent diffusion model with the following command:
  ```bash
  bash launcher/train_ldm.sh
  ```
  If you want to include depth information, you could add `--include_depth` to the command in the `train_ldm.sh` file.

- Then you could generate the goal image with the following command:
  ```bash
  python inference_ldm_goal_image.py --ckpt_folder lavis/output/LDM/runs/pix2pix (--include_depth)
  ```
  The results will be saved in the `lavis/output/LDM/results` folder.

## Citation
```
@article{zhen20243dvla,
  author = {Zhen, Haoyu and Qiu, Xiaowen and Chen, Peihao and Yang, Jincheng and Yan, Xin and Du, Yilun and Hong, Yining and Gan, Chuang},
  title = {3D-VLA: 3D Vision-Language-Action Generative World Model},
  journal = {arXiv preprint arXiv:2403.09631},
  year = {2024},
}
```

## Acknowledgement
Here we would like to thank the following resources for their great work:
- [SAM](https://github.com/facebookresearch/segment-anything), [ConceptFusion](https://github.com/concept-fusion/concept-fusion) and [3D-CLR](https://github.com/evelinehong/3D-CLR-Official) for Data Processing.
- [Diffusers](https://github.com/huggingface/diffusers), [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix), [StableDiffusion](https://github.com/Stability-AI/StableDiffusion) and [Point-E](https://github.com/openai/point-e) for the Diffusion Model.
- [LAVIS](https://github.com/salesforce/LAVIS) and [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM) for the Codebase and Architecture.
- [OpenX](https://robotics-transformer-x.github.io) for Dataset.
- [RLBench](https://github.com/stepjam/RLBench) and (Hiveformer)[https://github.com/vlc-robot/hiveformer] for Evaluation.