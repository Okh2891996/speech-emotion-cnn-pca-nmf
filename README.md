# Cross-Corpus Speech-Emotion Recognition  
MFCC ➜ PCA/NMF Compression ➜ CNN & SVM Benchmarks

> *Reproducible EE8104 Adaptive Signal Processing project (Winter 2025) Supervisor Prof Dr. Sridhar Krishnan*

Speech-emotion recognition (SER) models often perform brilliantly on a **single dataset** yet stumble the moment recording conditions—or speakers—change.  
This repo tackles that mismatch by **training and evaluating on two publicly available corpora at once**:

* **TESS** — Toronto Emotional Speech Set (2 800 clips)  
* **RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song (1 260 clips)

Our pipeline walks through every stage, end-to-end:

1. **Feature extraction** – 13 × 85 MFCC maps for each utterance.  
2. **Low-rank analysis** – PCA & NMF reveal shared structure and let us compress features to 26 / 52 dimensions.  
3. **Classical baseline** – RBF-SVMs on the compressed vectors (48 – 65 % accuracy).  
4. **Deep baselines** –  
   * A 5-block CNN on full MFCC maps (≈ 91 % test accuracy)  
   * A dual-input CNN that fuses MFCC maps with 26-D PCA+NMF vectors (≈ 92 % test accuracy).  
5. **Learning-curve study** – How data volume affects SVM vs. CNN performance.



|  #  | Notebook                             | 1-line tagline                                                      | Details & key outputs                                                                                                                                                                                                 |
| :-: | ------------------------------------ | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  01 | **`MFCC_TESS_NOTEBOOK.ipynb`**       | Generate 13 × 85 MFCC maps for the 2 800 TESS recordings.           | • Loads TESS wav files → 16 kHz → 0.5 s pads/trims → MFCC<br>• Saves `X_tess.npy`, `y_tess.npy`, and 80/20 split indices.<br>• Quick sanity MLP reaches \~100 % (shows dataset is easy to over-fit).                  |
|  02 | **`MFCC_RAVDESS_NOTEBOOK.ipynb`**    | Same MFCC pipeline for the 1 260 RAVDESS speech clips.              | • Produces `X_ravdess.npy`, `y_ravdess.npy`.<br>• Sanity MLP tops out at \~80 %.                                                                                                                                      |
|  03 | **`PCA and NMF COM.ipynb`**          | Run PCA + NMF on the pooled corpus and plot the bases.              | • Concatenates both MFCC sets.<br>• Computes first 6 PCA & NMF components.<br>• Saves scatter plot (PCA) and bar chart (NMF) used in Figs 2–3.<br>• Dumps `pca_com.npy`, `nmf_com.npy` for later feature compression. |
|  04 | **`NMF PCA TESS.ipynb`**             | Stand-alone PCA/NMF exploration for TESS only.                      | • Generates TESS-specific panels for Fig 2/3.                                                                                                                                                                         |
|  05 | **`NMF PCA RAVESS.ipynb`**           | Stand-alone PCA/NMF exploration for RAVDESS only.                   | • Generates RAVDESS-specific panels for Fig 2/3.                                                                                                                                                                      |
|  06 | **`SVM_NFM_PCa.ipynb`**              | Train SVM baselines on compressed (PCA + NMF) features.             | • Builds 26-dim (2 × 2) **and** 52-dim (4 × 4) vectors.<br>• LOOCV + hold-out test:<br>  • 2 × 2 → \~0.48 test acc<br>  • 4 × 4 → \~0.65 test acc<br>• Saves 4 × 4 confusion matrix for the appendix.                 |
|  07 | **`CNN_COMBINEdata_Gaussian.ipynb`** | Baseline CNN on full MFCC maps with Gaussian-noise augmentation.    | • Doubles training data with σ = 0.01 noise.<br>• 5-block CNN, 50 epochs → \~0.91 test acc.<br>• 7 × 7 confusion matrix = Fig 1; per-class PR table.                                                                  |
|  08 | **`CNN_PCA_NMF.ipynb`**              | Lightweight two-input CNN that consumes 26-dim compressed features. | • Dual branch: MFCC map & 26-dim side vector.<br>• \~0.92 test acc (numbers in Table VI).                                                                                                                             |
|  09 | **`ML vs DL.ipynb`**                 | Compare learning curves of SVM vs. CNN as data size grows.          | • Trains both models on 10 – 100 % of data.<br>• Plots Figure 4 and prints the CSV behind it.                                                                                                                         |


PDF for details: [Deep Learning Vs Machine Learning A Study on SER.pdf](https://github.com/user-attachments/files/20856341/Deep.Learning.Vs.Machine.Learning.A.Study.on.SER.pdf)
