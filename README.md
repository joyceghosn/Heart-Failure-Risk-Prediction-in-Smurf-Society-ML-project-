\section*{Project Overview}

This project develops a complete machine learning pipeline to predict \textbf{10-year heart failure risk} in Smurf society using \textbf{clinical tabular data} and \textbf{heart-scan image features}.  
The system is built in four parts: preprocessing, linear modeling, nonlinear modeling, multimodal modeling with CNN embeddings, and interpretability.
The dataset contains clinical biomarkers, lifestyle indicators, demographic variables, and MRI-like heart images.  
The goal is to build accurate predictive models and identify key determinants of cardiovascular vulnerability.

\section*{Repository Structure}

\begin{verbatim}
.
├── data/        # Sample raw & processed data
├── notebooks/   # Exploration, training, evaluation notebooks
├── src/         # Preprocessing & model scripts (optional)
├── outputs/     # Figures, tables, predictions
├── docs/        # Report, slides, diagrams
└── README.md
\end{verbatim}

\section*{Part 1: Data Preparation and Linear Model}

\subsection*{Preprocessing}
Based on the original report, 14 tabular features were processed using:
\begin{itemize}[noitemsep]
    \item \textbf{ColumnTransformer}: ordinal encoding, one-hot encoding.
    \item \textbf{StandardScaler} for linear model stability.
    \item Strict split: 80\% inner-train, 20\% inner-validation, 500-row test set untouched.
\end{itemize}

\subsection*{Feature Selection}
Pearson correlation retained 13 predictive features.

\subsection*{Linear Model Results}
Ridge Regression (\(\alpha = 100\)) performed best:
\begin{itemize}[noitemsep]
    \item Validation RMSE: 0.0506
    \item Test RMSE: 0.05596
\end{itemize}

\section*{Part 2: Nonlinear Models and Feature Selection}

Models evaluated:
\begin{itemize}[noitemsep]
    \item Random Forest
    \item Gradient Boosting
    \item k-NN
    \item MLP Neural Network
\end{itemize}

Feature selection techniques:
\begin{itemize}[noitemsep]
    \item Mutual Information
    \item Sequential Forward Selection (SFS + MLP)
    \item Random Forest Importances
\end{itemize}

Hyperparameter tuning via 5-fold \texttt{GridSearchCV}.

\subsection*{Best Model}
\textbf{Gradient Boosting (all 18 features)}:
\begin{itemize}[noitemsep]
    \item Validation RMSE: 0.0431
    \item Test RMSE: 0.04191
\end{itemize}

\section*{Part 3: Multimodal Model (Images + Tabular Data)}

\subsection*{CNN Feature Extraction}
A custom CNN extracts an 8-dimensional embedding per heart image, representing high-level structural patterns.

\subsection*{Multimodal Dataset}
\begin{itemize}[noitemsep]
    \item 18 tabular features
    \item 8 image embeddings
    \item Total: 26 predictors
\end{itemize}

Mutual Information was applied to select the \textbf{Top-20 multimodal features}.

\subsection*{Best Multimodal Model}
Gradient Boosting with MI-Top-20 features:
\begin{itemize}[noitemsep]
    \item Validation RMSE: 0.02797
    \item Test RMSE: 0.03041
\end{itemize}

This represents a \textbf{27.4\% improvement} over the tabular-only Gradient Boosting model.

\section*{Part 4: Understanding Heart Failure Risk}

Visual analyses support the following hypotheses:

\begin{enumerate}[noitemsep]
    \item Age strongly increases risk.
    \item High blood pressure + high cholesterol defines the most vulnerable subgroup.
    \item Unhealthy lifestyle (liquor, donuts) increases risk.
    \item Certain professions (administration, services) show higher baseline risk.
    \item A clear combined high-risk profile emerges: older, hypertensive, hypercholesterolemic, unhealthy diet.
\end{enumerate}

\section*{Final Model Comparison}

\begin{center}
\begin{tabular}{|c|c|}
\hline
\textbf{Model} & \textbf{Test RMSE} \\
\hline
Ridge (linear) & 0.05596 \\
Gradient Boosting (tabular) & 0.04191 \\
\textbf{Multimodal Gradient Boosting} & \textbf{0.03041} \\
\hline
\end{tabular}
\end{center}

\section*{How to Run}

\subsection*{Install Requirements}
\begin{verbatim}
pip install -r requirements.txt
\end{verbatim}

\subsection*{Run Preprocessing}
\begin{verbatim}
python src/preprocess.py
\end{verbatim}

\subsection*{Train Models}
\begin{verbatim}
python src/train_model.py
python src/train_multimodal.py
\end{verbatim}

\subsection*{Evaluate}
\begin{verbatim}
python src/evaluate.py
\end{verbatim}

\section*{Tech Stack}
\begin{itemize}[noitemsep]
    \item Python
    \item pandas, numpy
    \item scikit-learn (Ridge, GBM, RF)
    \item TensorFlow/Keras (CNN)
    \item Matplotlib, Seaborn
\end{itemize}

\end{document}
