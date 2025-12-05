# CUT_WGAN Model Trainer Readme

This repository implements the training process for a **(CUT) Contrastive Unpaired Translation**  with **(WGAN) Wasserstein Generative Adversarial Network** model, However, we are getting smooth images due to which patterns cannot be observed.
We have implemented CUT_With_GAN_LOSS as well through which we got much better results. for that switch to branch **CUT_with_GAN_loss**
Likely for an image-to-image translation task such as **Super-Resolution** (translating low-resolution images to high-resolution ones, as suggested by the `PSNR` metric).

---

### 💻 Main Execution File

The primary script responsible for running the training process is:

* **`CUTtrain.py`**: This file contains the main **`CUTTrainer`** class and the entry point (`if __name__ == "__main__":`). It handles data loading, model initialization, the complete training loop, loss calculation, PSNR tracking, and saving the results.

---

#Deep-Learning

# Dependencies and Project Structure
The project relies on a standard set of Python packages and several custom modules organized into dedicated directories.
# Python Libraries

The following standard libraries are required (installable via `pip`):
| Dependency | Purpose |
| :--- | :--- |
| **`torch`** | Core PyTorch library for tensor operations and deep learning. |
| **`torchvision`** | PyTorch package for computer vision, including transforms and image utilities. |
| **`matplotlib`** | Used for generating the **PSNR curve plot** saved in the `results` directory. |
| **`argparse`** | Used in `CUTtrain.py` to parse command-line arguments for training configuration. |

# Custom Modules (Local Dependencies)

The `CUTtrain.py` file requires the existence and correct placement of the following custom directories and files, which define the model architecture, data handling, and specific losses:

| Directory/File | Key Files Used | Purpose in `CUTtrain.py` |
| :--- | :--- | :--- |
| **`Models/`** | `CUT.py`, `Network.py` | Defines the **`CUTModel`** (Generator and Discriminator) and fundamental network components. |
| **`Losses/`** | `NCE_losses.py` | Defines the **PatchNCE loss** (Contrastive Loss) critical to the CUT architecture. |
| **`Utils/`** | `dataset.py`, `transformers.py` | Contains the **`IRImageDataset`** for data loading and the **`transform_pipeline`** for preprocessing. |
| **`checkpoints/`** | (Directory) | Likely used to store the saved model weights during or after training. |
| **`results/`** | (Directory) | The output directory where images, PSNR logs, and plots are saved after training. |

---

#How to Run

To start the training, execute the main file from the root directory (`~/Deep-Learning`) and specify the data paths and training parameters:
```bash
**python3 CUTtrain.py**
