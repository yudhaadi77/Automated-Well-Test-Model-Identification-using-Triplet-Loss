# Well Test Model Identification using Triplet Loss

This project applies deep metric learning to automate the identification of well test responses using log-log derivative plots. The goal is to reduce manual interpretation in Pressure Transient Analysis (PTA) by retrieving the most visually similar cases from a reference database.

## 🔍 Objective

To support engineers in well test interpretation by enabling faster and more consistent identification of reservoir models such as Homogeneous and Radial Composite, based on image similarity.

## 🧠 Methodology

- **Learning type:** Deep metric learning using Triplet Loss
- **Backbone:** MobileNetV2 (pretrained)
- **Output:** 128-dimensional embedding space
- **Training:** 1000 triplets over 10 epochs
- **Evaluation:** Top-K similarity retrieval and t-SNE visualization

## 🧰 Workflow

1. Synthetic images (Loglog, Semilog, History plots) generated from known PTA models
2. Images resized to 224×224, normalized to [0, 1]
3. Anchor–positive–negative triplets created based on class
4. Embedding model trained using Triplet Loss
5. Top-K retrieval and t-SNE visualization used for evaluation

## 📂 Dataset

- Total images: ~150  
- Two classes: `Homogeneous`, `Radial Composite`  
- Each case includes:
  - Loglog plot  
  - Semilog plot  
  - History plot  
- Dataset is stored in the `dataset/` folder of this repository  
- File format: `.png`

> **Note:** The dataset could not be uploaded to Fataiku due to repeated errors. It is included directly in this GitHub repository.

## 📈 Results

- **Top-1 accuracy:** 94%  
- **Top-3 accuracy:** 100%  
- **t-SNE visualization:** Clear class separation between Homogeneous and Radial Composite

## 📦 Dependencies
- TensorFlow
- NumPy
- scikit-learn
- matplotlib
- Pillow

## 📌 Future Work
- Add more classes to the dataset (e.g., Composite, Faulted, Fractured models)
- Expand the number of cases per class to improve generalization
- Evaluate model performance using real field test data
- Build a lightweight API or web-based demo for practical use

## 📚 References

- Nukala, S. T., et al. (2024). Enhancing Pressure Transient Analysis with Automatic Model Identification: A Machine Learning Approach. SPE-218836-MS. DOI: 10.2118/218836-MS
- Yan, R. (2023). Well Test Model Identification using Deep Similarity Learning. Undergraduate Thesis, Institut Teknologi Bandung.

## 👥 Contributors
- Yudha (yudhaadiputra26@gmail.com)
- Sherly
- Arvin
  
## 🚀 Example: Predicting Similar Images

```python
top_paths = predict_nearest(
    query_path="dataset/Homogenous/Homogenous_3_Loglog_plot.png",
    embedding_model=embedding_model,
    reference_embeddings=all_embeddings,
    reference_paths=image_paths,
    reference_labels=labels,
    top_k=5
)
