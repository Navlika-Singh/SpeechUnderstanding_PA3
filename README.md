# [Speech Understanding] Programming Assignment 3

This readme file contains detailed explanation of the code provided and instructions on how to run each file for successful reproduction of the reported results.

#### Environment requirements
First, note that in order for the provided code to run successfully, an environment with all the required packages must be installed. Consequently, I have provided a *'package_requirements.txt'* file which contains the name and respective version of all the packages used during this programming assignment and are necessary to reproduce the results.

Second, note that the provided code is computationally and storage-wise extensive. This implies that it requires strong GPU-resources and large storage (for data, models etc.). Consequently, please run the provided code files on such a system only.

#### Pre-requisites
Now, note that there are data-wise and model-wise pre-requisites of the code. This is listed as follows:

##### Data-wise pre-requisites
Now, the required data must be downloaded and located in the specified path.
 - 'Custom' dataset provided in the instructions.
 - 'FOR' dataset provided in the instructions.

##### Model-wise pre-requisites
Now, the required model checkpoints must be downloaded and located in the specific path.
 - SSL W2V model trained for LA and DF tracks of the ASVSpoof   dataset.

Please note that each model has it package requirements provided on its official github repository (requirements.txt) which should be loaded for its successful execution. 

### Instructions

Now, note the following set of instructions for the successful reproduction of provided results.

- Run the main.py file with the appropriate paths for the dataset and pre-trained models to reproduce the results of task 1.
- Run the main_finetune.py file with the appropriate paths to finetune the pre-trained models on the FOR dataset.
- Run the main.py file with the appropriate paths for the dataset and fine-tuned models to reproduce the results of task 3.
- Run the gradio_run.py file with appropriate paths to produce the gradio link for the trained model to evaluate the trained model on the browser.

NOTE: Please contact the author in case of any discrepancy.