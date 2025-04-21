# 🧠 Deep Learning Project 2: News Classification with RoBERTa + LoRA + Knowledge Distillation

This project fine-tunes a RoBERTa-based model on the [AG News dataset](https://huggingface.co/datasets/ag_news) for multi-class news classification. We explore performance and efficiency by combining **LoRA** (Low-Rank Adaptation) and **Knowledge Distillation** to train a lightweight student model from a larger teacher model.

---

## 🚀 Project Highlights

- ✅ Fine-tunes **RoBERTa-base** on AG News (4 categories)
- ✅ Implements **text cleaning** and **random deletion** for augmentation
- ✅ Uses **LoRA** to reduce student model trainable parameters (<1M)
- ✅ Applies **knowledge distillation** to transfer knowledge from a teacher model
- ✅ Custom `DistillationTrainer` with KL Divergence + Cross-Entropy loss
- ✅ Final model tested on unseen data and exported as `submission.csv`

---

## 🗂️ Repository Structure

```
deep-learning-project-2/
├── script.py                # Main training script
├── submission.csv           # Output predictions on test set
├── README.md                # Project overview and instructions
├── final_student_model/     # Final fine-tuned student model (optional)
├── teacher_model/           # Saved teacher model checkpoints
├── student_model/           # Saved student model (distilled)
```

---

## 📊 Dataset

- **Name:** AG News
- **Source:** [HuggingFace Datasets](https://huggingface.co/datasets/ag_news)
- **Classes:** World, Sports, Business, Sci/Tech
- **Test Data:** Unlabeled test set in `test_unlabelled.pkl` (provided externally)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/deep-learning-project-2.git
cd deep-learning-project-2
```

### 2. Install Dependencies

You need Python 3.8+ and the following libraries:

```bash
pip install torch transformers datasets scikit-learn peft
```

If using a GPU-enabled environment (e.g., Kaggle, Colab, or local CUDA), make sure `torch` is installed with GPU support.

---

## 🧪 Training Pipeline

### Step 1: Preprocess & Augment

- Clean text (remove URLs, extra spaces)
- Apply random deletion for augmentation (training set only)
- Tokenize using `roberta-base` tokenizer

### Step 2: Train the Teacher Model

- Full fine-tuning of RoBERTa-base
- No parameter efficiency constraints

### Step 3: Train the Student with Distillation

- Use **LoRA** to reduce trainable parameters
- Distill knowledge using a custom loss:
  \[
  	ext{Loss} = lpha \cdot 	ext{KL}(S \| T) + (1 - lpha) \cdot 	ext{CE}
  \]

### Step 4: Final Fine-Tuning

- Fine-tune the distilled student on the clean, full training set

### Step 5: Predict on Test Set

- Clean and tokenize unlabeled test data
- Generate predictions and export `submission.csv`

---

## 🔧 Configuration

Modify hyperparameters like learning rate, batch size, epochs, and LoRA config in the script as needed:

```python
LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    ...
)
```

---

## 📈 Results

| Model Type      | Validation Accuracy |
|-----------------|---------------------|
| Teacher (RoBERTa-base) | ~**94.56%**            |
| Student (LoRA + Distillation) | ~**94.78%**            |


---

## 📦 Output

After training, the final student model generates a `submission.csv`:

```csv
ID,label
0,2
1,0
2,1
...
```

---

## ✨ Future Improvements

- Try different LoRA configurations (e.g., higher `r`)
- Explore alternative student architectures (e.g., DistilRoBERTa)
- Use Mixup or backtranslation for more aggressive augmentation
- Integrate TensorBoard or Weights & Biases for experiment tracking

---

## 👨‍💻 Authors

- **Your Name** – [@yourhandle](https://github.com/yourhandle)

> Deep Learning Project 2 — Spring 2025  
> Course Instructor: [Instructor Name]

---

## 📜 License

This project is open-sourced for educational use. See [LICENSE](LICENSE) for details.
