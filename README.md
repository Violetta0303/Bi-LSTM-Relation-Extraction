# README: SemEval-2010 Task 8 Relation Extraction Project

## Project Background

This project focuses on the relation extraction task from the SemEval-2010 Task 8 dataset. The objective is to identify and classify semantic relationships between entities in natural language text. The task requires classifying entity pairs into 19 relationship categories, consisting of 18 specific relationships and one "Other" class. A two-stage classification approach is adopted:

1. **Stage 1**: A logistic regression model (binary_classifier.pkl) distinguishes between the "Other" class and "Non-Other" classes.
2. **Stage 2**: For samples identified as "Non-Other," a BiLSTM model with an attention mechanism (attention_bilstm_best.pt) classifies them into one of the 18 specific relationships.

The project encompasses feature engineering, data standardization, model training, and evaluation, with visualizations to demonstrate the training process and performance outcomes.

------

## Environment Requirements

To run this project, the following software and libraries are required:

- **Python**: Version 3.6 or higher
- **PyTorch**: Version 1.7 or higher (with optional GPU or MPS support)
- **scikit-learn**: For logistic regression and data standardization
- **SpaCy**: For feature extraction
- **NLTK**: For text processing
- **Matplotlib** and **Seaborn**: For generating visualizations
- **Datasets**: For loading the SemEval-2010 Task 8 dataset
- **Tqdm**: For displaying training progress

------

## Installation Guide

1. Install Dependencies:

   Execute the following command in the project root directory to install the required Python libraries:

   ```bash
   pip install torch scikit-learn spacy nltk matplotlib seaborn datasets tqdm
   ```

4. Download and Install SpaCy English Model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

6. Download GloVe Word Vectors:

   - Download glove.6B.300d.txt (or another version) from the [GloVe website](https://nlp.stanford.edu/projects/glove/).
   - Place the file in the project root directory or update the code with the correct file path.

------

## Data Preparation

1. Load Dataset:

   The SemEval-2010 Task 8 dataset is utilized and can be loaded using the datasets library:

   ```python
   from datasets import load_dataset dataset = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")
   ```

3. Data Augmentation (Optional):

   - The data.csv file in the project directory contains an augmented dataset (if provided). Ensure its format aligns with the original dataset, including sentences, entity markers, and relationship labels.

------

## Model Training

1. Run the Training Script:

   - Open train_eval.ipynb and execute the cells sequentially.
   - This script manages data preprocessing, feature extraction, model training, and evaluation.

2. Training Outputs:

   - **Feature Extractor**: Saved as feature_extractor.pkl
   - **Scaler**: Saved as scaler.pkl
   - **Binary Classifier**: Saved as binary_classifier.pkl
   - **BiLSTM Model**: Best model saved as attention_bilstm_best.pt
   - **Visualization**: Training and validation metrics saved as training_validation_metrics.png

------

## Model Evaluation

1. Run the Test Script:

   - Open Test.ipynb and execute the cells to load the saved models and evaluate them on the test data.

2. Evaluation Outputs:

   - **Stage 1**: Accuracy, F1 score, and recall for the binary classifier.
   - **Stage 2**: Macro F1 score, accuracy, and Macro recall for the 18-class relationship classification.
   - **Confusion Matrix**: Saved as confusion_matrix.png.

------

## Project Structure

The project directory includes the following files:

- **attention_bilstm_best.pt**: Saved BiLSTM model for the second-stage 18-class relationship classification.
- **binary_classifier.pkl**: Saved logistic regression model for the first-stage "Other" vs. "Non-Other" classification.
- **confusion_matrix.png**: Confusion matrix image illustrating the model’s performance on the test set.
- **data.csv**: Dataset file (optional) for training or testing.
- **feature_extractor.pkl**: Feature extractor for converting raw text into model inputs.
- **scaler.pkl**: Scaler for standardizing features.
- **Test.ipynb**: Test script containing model loading and evaluation code.
- **train_eval.ipynb**: Training and evaluation script containing data processing and model training code.
- **training_validation_metrics.png**: Image displaying training and validation metrics (e.g., loss and accuracy curves).

------

## Notes

- **GloVe Path**: Verify that the path to glove.6B.300d.txt is correctly specified in the code.
- **Device Compatibility**: For MPS devices (e.g., Apple Silicon), ensure your PyTorch version supports MPS.
- **Data Augmentation File**: If using data.csv for augmentation, confirm it matches the SemEval-2010 Task 8 dataset format.
- **Duplicate Files**: Check for duplicate Test.ipynb files in the directory and remove them to avoid confusion.
- Glove Download：https://drive.google.com/file/d/1P85bAjABywwEK5QJnc_VDKKgBJr6_Wef/view?usp=drive_link

  attention_bilstm_best.pt:

  https://drive.google.com/file/d/1loeV3RD11GT_d--FWyhP6rlNujgl4gJd/view?usp=drivesdk

  binary_classifier.pkl:

  https://drive.google.com/file/d/14r-r39cfU9B_uvUmK1t-t38Cb9gJgk9C/view?usp=drive_link

  feature_extractor:

  https://drive.google.com/file/d/1W6NhQkSLeGzu3sx0XCdeuZWGAEZn0LSG/view?usp=drive_link

  scaler:

  https://drive.google.com/file/d/1ikKYQu0ODVIp9EYeRoqCaNsq0iwW6fb1/view?usp=drive_link

  Data.csv:

  https://drive.google.com/file/d/1it9Kp-EL2rSk2-_9b4cIcjmu0VLKe59g/view?usp=drive_link

------

## Contributions

Contributions are welcome through issues or pull requests. For significant changes, please open an issue first to discuss the proposed updates.

## License

This project is licensed under the [MIT License](LICENSE).
