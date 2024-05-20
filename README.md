# Image-to-Text Generation with VisionEncoderDecoderModel

This project aims to train the VisionEncoderDecoderModel for image-to-text generation, specifically using the BEiT model as the image encoder and GPT-2 as the text decoder. 

## Project Structure

```plaintext
project/
│
├── src/
│   ├── beit-model.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── utils.py
│
├── main.py
├── inference.py
└── requirements.txt
```
Files and Directories
*    src/beit-model.ipynb: Notebook for experiments
*    src/config.py: Configuration settings for the project.
*    src/dataset.py: Dataset and data collator definitions.
*    src/model.py: Model setup and configuration.
*    src/trainer.py: Custom trainer class and training arguments.
*    src/utils.py: Utility functions.
*    ./main.py: Main script to run the training and 
*    ./inference.py: Script for testing models on images.
*    ./requirements.txt: List of dependencies.


## Setup

1. Clone the repository:
    ```shell
        git clone https://github.com/yourusername/project.git
        cd project
    ```

2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Prepare the dataset
    Make sure you have the COCO 2017 dataset downloaded and available. Update the paths in src/config.py 

4. Update Config
    Update config.py hyperparameters for training

## Usage
### Training the Model
Run the main.py script to start training the model.
    ```python main.py```
### Inference

Run the inference.py script to use trained models
     ```python inference.py```
    