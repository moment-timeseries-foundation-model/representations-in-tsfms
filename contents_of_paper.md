\begin{abstract}
Time series foundation models (TSFMs) promise to be powerful tools for a wide range of applications. However, their internal representations and learned concepts are still not well understood. In this study, we investigate the structure and redundancy of representations across various TSFMs, examining the self-similarity of model layers within and across different model sizes. This analysis reveals block-like redundancy in the representations, which can be utilized for informed pruning to improve inference speed and efficiency. Additionally, we explore the concepts learned by these models—such as periodicity and trends—and how these can be manipulated through latent space steering to influence model behavior. Our experiments show that steering interventions can introduce new features, e.g., adding periodicity or trends to signals that initially lacked them. These findings underscore the value of representational analysis for optimizing models and demonstrate how conceptual steering offers new possibilities for more controlled and efficient time series analysis with TSFMs.
\end{abstract}
%\section{Probing and Intervening in Time Series Foundation Models}
\section{Methods}
We study TSFMs using three complementary methods of analysis and interventions concerning model layer-wise representations. In Section~\ref{sec:representational_analysis_pruning}, we examine learned representations through the lens of similarity, uncovering stored knowledge redundancy in TSFM representations. We leverage this redundancy to prune multiple layers of pre-trained models, thereby improving their efficiency. In Section~\ref{sec:localize_concepts}, we identify specific concepts learned by TSFMs and localize them to specific parts of model layers and tokens. In Section~\ref{sec:steer_concepts}, we explore the ability to steer model predictions by intervening in model hidden representations across layers, enabling us to guide model output toward concept-informed predictions.

\subsection{Representation Similarity Analysis and Pruning}
\label{sec:representational_analysis_pruning}
%\subsubsection{Analyzing Representational Similarity for Effective Pruning}

To gain a deeper understanding of TSFM representations, we investigate the following research questions: \textit{(RQ1)} How similar are the representations learned by models of the same size but belonging to different families? \textit{(RQ2)} How do these representations differ across models of varying sizes within the same family? \textit{(RQ3)} How similar are the representations learned by corresponding layers of different TSFMs within the same family? 

We consider several metrics commonly employed in the literature to analyze the similarity between learned representations. While our primary analysis relies on Centered Kernel Alignment (CKA)~\cite{kornblith2019similarity}, we also explored Cosine Similarity and Singular Vector Canonical Correlation Analysis (SVCCA)~\citep{raghu2017svcca}. For brevity, we provide a brief overview of CKA below, while detailed descriptions of the remaining metrics can be found in Appendix~\ref{app:additional_representation_similarity_metrics}.

\paragraph{Representational Similarity using CKA.} CKA measures the similarity of representations by comparing the centered kernel matrices. It has been shown to be effective in capturing similarities between layers of deep networks~\citep{kornblith2019similarity}. The general form of CKA between two sets of representations $\mathbf{X}$ and $\mathbf{Y}$ is defined as:
\begin{equation}
\text{CKA}(\mathbf{X}, \mathbf{Y}) = \frac{\text{HSIC}(\mathbf{X}, \mathbf{Y})}{\sqrt{\text{HSIC}(\mathbf{X}, \mathbf{X}) \cdot \text{HSIC}(\mathbf{Y}, \mathbf{Y})}}
\end{equation}
where $\text{HSIC}$ denotes the Hilbert-Schmidt Independence Criterion \cite{gretton2005measuring}. 

\begin{figure}
 \centering
 \includegraphics[clip, trim=0 150 795 0, width=0.5\textwidth]{iclr2025/figures/pruning_methods.pdf}
 \caption{
    For each identified block of layers exhibiting redundant representations (red), we remove the internal layers of the block by zeroing out their weights (blue). For example, if a block consists of five layers, we prune layers 2 through 4, retaining only the first and last layers to reduce representation redundancy while maintaining model integrity. More details on pruning can be found in App.~\ref{app:redundant_blocks}.
    }
\label{fig:pruning_methods}
\end{figure}

For computational efficiency, we utilized a linear kernel in our CKA calculations, resulting in the following simplified formula:
\begin{equation}
\text{CKA}_{\text{linear}}(\mathbf{X}, \mathbf{Y}) = \frac{\|\mathbf{X}^T \mathbf{Y}\|_F^2}{\|\mathbf{X}^T \mathbf{X}\|_F \cdot \|\mathbf{Y}^T \mathbf{Y}\|_F}
\label{eq:linear_cka}
\end{equation}
where $\|\cdot\|_F$ denotes the Frobenius norm. The denominator in Equation~\ref{eq:linear_cka} ensures that the metric value falls within the range of 0 to 1, facilitating interpretability. A high CKA value indicates a strong alignment between the two sets of representations, suggesting that the layers are likely learning similar features or concepts.

\paragraph{Pruning TSFMs Based on Representational Similarity.} Large TSFMs typically learn redundant representations, which often manifest as block-like structures in heatmaps depicting pairwise similarity between layer activations (Figure \ref{fig:pruning_methods}). We can leverage this redundancy to prune TSFMs, enhancing inference speed while preserving accuracy. We build on prior work \citep{nguyen2021wide} and propose a simple layer pruning strategy, which we call \textit{Block-wise Pruning}, outlined in Algorithm~\ref{alg:redundant_pruning}. To preserve the structural integrity of each block, we retain the first and last layers of each block while zeroing out the weights of the intermediate layers. The skip connections within transformer blocks ensure that signals and gradients continue to flow through the rest of the network.

\begin{algorithm}[t!]
\caption{Block-wise Pruning}
\label{alg:redundant_pruning}
\begin{algorithmic}
\REQUIRE Trained model $\mathcal{M}$ with layers $\{l_1, l_2, \dots, l_n\}$; Identified redundant blocks $\mathcal{B} = \{b_1, b_2, \dots, b_k\}$
\FOR{each block $b_i$ in $\mathcal{B}$}
    \STATE Let $b_i$ consist of layers $l_s$ to $l_e$ \COMMENT{Block edges at $l_s$ and $l_e$}
    \FOR{layer index $j = s+1$ to $e-1$}
        \STATE Zero out the weights of layer $l_j$ in model $\mathcal{M}$
    \ENDFOR
\ENDFOR
\STATE \textbf{return} $\mathcal{M}'$
\end{algorithmic}
\end{algorithm}

%\subsubsection{Research Questions and Experimental Setup}
% To gain a deeper understanding of TSFM representations, we investigate the following research questions: \textit{(RQ1)} How similar are the representations learned by models of the same size but belonging to different families? \textit{(RQ2)} How do these representations differ across models of varying sizes within the same family? \textit{(RQ3)} How similar are the representations learned by corresponding layers of different TSFMs within the same family? To answer these questions, we use Centered Kernel Alignment (CKA) to measure the similarity between representations at different layers of TSFMs, and visualize the results using heatmaps.

To demonstrate the effectiveness of our proposed pruning strategy, we explore two pruning configurations, one in which we prune all redundant blocks, and the other where prune only a single block. We compare the performance of these pruned models to the original, unpruned TSFMs using standard task-specific accuracy metrics (Mean Squared Error and Mean Absolute Error) and efficiency metrics (inference time in milliseconds and theoretical model size in megabytes). We evaluate these models on widely used imputation~\citep{Informer} and forecasting~\citep{ansari2024chronos} benchmarks in both zero-shot settings and after linear probing \citep{goswami2024moment}. 

\begin{figure*}[!htb]
\centering
 \includegraphics[width=0.95\textwidth]{iclr2025/figures/Pruning_Steering_Methods.pdf}
 \caption{\textbf{Overview of linear probing, concept localization, and steering.} During linear probing we train linear classifiers $f_{ij}(\mathbf{h}_i^{(j)}, \theta_i^{(j)})$ to classify time series $\mathbf{x}$ into constant $c$ and sinusoid $s$ classes, using hidden representations $\mathbf{h}_i^{(j)}$ at the $i$-th layer and $j$-th token. We localize concepts using Fisher's Linear Discriminant Ratio (LDR) between the classes at each layer and token.
 % using mean and variance statistics of $h_i^{(j)}$ for each predicted class, $\hat{y}$. The LDR output is scaled between 0 and 1 using min-max scaling to allow for consistent comparison across layers. 
 The concept steering vector at the $i$-th layer is defined as the difference between the median activation matrices of the sinusoid and constant time series classes, \(\mathbf{M}_{i_s} - \mathbf{M}_{i_c}\). Vectors of all layers are stacked into a steering matrix $\mathbf{S}$ to steer model predictions towards desired concepts by updating the embeddings as $\mathbf{h}_i \leftarrow \mathbf{h}_i + \lambda \mathbf{S}_i$, where $\lambda$ is a scalar that controls the strength of the intervention.}
\label{fig:concept_representation_methods}
\end{figure*}


% \subsection{Probing and Intervening in Time Series Foundation Models}
% \label{sec:concept_discovery_and_steering}


\begin{figure}[!thb]
\centering
\setlength{\tabcolsep}{0pt}
\begin{tabular}{c}
    \includegraphics[width=0.49\textwidth, trim={0cm 1cm 0cm 0cm}, clip]{icml2025_arxiv/figures/time_series_examples/trend.pdf} \\
    \includegraphics[width=0.49\textwidth, trim={0cm 1cm 0cm 0cm}, clip]{icml2025_arxiv/figures/time_series_examples/sine.pdf} \\
    \includegraphics[width=0.49\textwidth, trim={0cm 0.35cm 0cm 0cm}, clip]{icml2025_arxiv/figures/time_series_examples/sine_and_trend.pdf} \\
\end{tabular}
\caption{Examples of synthetic data generated for concept localization, and concept steering experiments include constant signals with varying trend (top), sinusoidal signals with varying frequency (middle), and compositions of constant signals with varying trends and sinusoidal signals, resulting in sinusoidal signals with varying trends (bottom).}
     \label{fig:synthetic_data_examples_main}
\end{figure}




\subsection{Identifying and Localizing Time Series Concepts}\label{sec:localize_concepts}

Through our experiments, we aim to answer the following research questions: \textit{(RQ4)} Do TSFMs represent concepts associated with specific data-generating functions distinctly in the latent space? \textit{(RQ5)} Are these learned concepts localized to specific layers and tokens within TSFMs? 

To systematically explore the ability of TSFMs to understand intuitive time series concepts, we randomly generate a large number of synthetic univariate time series. Each randomly generated time series belong to one of two pattern classes: \textit{constant} or \textit{sinusoidal}. Constant patterns, represented by $y(t) = m t + b$, capture long-term non-periodic trends. Sinusoidal patterns, modeled as $y(t) = a \sin\left( \frac{2\pi t}{f} \right)$, represent periodic processes. By controlling the parameters $m$, $b$, $a$, and $f$, we can systematically generate time series with varying slope, intercept, amplitude, and periodicity, respectively. Despite their simplicity, these data generation mechanisms capture a wide range of real-world time series patterns. Example series are shown in Fig.~\ref{fig:synthetic_data_examples_main}. For a detailed description of the data generation process, please refer to Appendix~\ref{app:synthetic_data_generation}.


\paragraph{Identifying Linearly Represented Features.} 
We build on the investigation approach outlined in \citep{marks2023geometry}. We say that a feature is linearly represented in a foundation model $\mathcal{M}$ if it is represented as a \textit{direction} in its latent space. If this feature is linearly represented in $\mathcal{M}$, we also want to identify which layer $l$ in $\mathcal{M}$ learns this concept in the most discriminant way. As a concrete example, consider that we want to check whether $\mathcal{M}$ can distinguish between constant and sinusoidal patterns.

To determine whether the feature (sinusoidal vs. constant time series) is linearly represented, we leverage the aforementioned synthetic dataset which comprises of multiple sinusoids and constant time series randomly sampled using our data generating function. Using this dataset, we extract  the residual stream of each transformer block after the feed-forward layer. Let $\mathbf{h}_i^{(j)} \in \mathbb{R}^{n \times D}$ denote the hidden representation of a time series $\mathbf{x}$ at $i$-th layer and $j$-th token of $\mathcal{M}$, where $D$ is the dimensionality of the hidden layer. Linear probing involves training separate linear models for each layer and token to classify time series $\mathbf{x}$ as a constant or sinusoid pattern. Classifiers $f_{ij}(\mathbf{h}_i^{(j)}, \theta_i^{(j)})$ are trained on the hidden representation $\mathbf{h}_i^{(j)}$ at each $i$-th layer and each $j$-th token to update the parameters $\theta_i^j$. Additionally, we perform probing on representations averaged along the token dimension for each $i$-th layer. The linear probes are trained to optimize the Fisher Criterion, a function that aims to maximize the distance between class means while minimizing within-class variance:
\begin{align}
\mathcal{L}_{\text{Fisher}}(c, s) &= -\frac{(\mu_s - \mu_c)^2}{\sigma_s^2 + \sigma_c^2}.
\end{align}
Here, $\mu_s$ and $\mu_c$ correspond to the mean embedding values, computed using all time series of a given class. Similarly, $\sigma^2_s$ and $\sigma^2_c$ correspond to the variance computed across the $n$ dimension for each class.

\paragraph{Localizing Linearly Represented Features.} 

To localize which layers and tokens learn a specific concept, we compute the Fisher's Linear Discriminant Ratio (LDR) between the classes using the mean and variance statistics of $\mathbf{h}_i^{(j)}$ for each predicted class $\hat{\mathbf{y}}$, which is determined using the classifier $f_{ij}$ during linear probing. The goal of LDR is to maximize the separation between the classes by comparing the variance $\sigma^2$ within each class to the difference between the class means, $\mu$. A larger ratio indicates a clearer separation between the two classes, which can aid in concept localization by identifying where the classes are well-separated in the feature space. When applied in the context of neural network activations, LDR helps highlight which layers or features are most discriminative
\begin{align}
    \text{LDR}(\mathbf{h}_i^{(j)}|\hat{\mathbf{y}}) &= \frac{(\mu_{\mathbf{h}_i^{(j)}|\hat{\mathbf{y}}=s} - \mu_{\mathbf{h}_i^{(j)}|\hat{\mathbf{y}}=c})^2}{\sigma_{\mathbf{h}_i^{(j)}|\hat{\mathbf{y}}=s}^2 + \sigma_{\mathbf{h}_i^{(j)}|\hat{\mathbf{y}}=c}^2} \\
    &= \frac{(\mu_s - \mu_c)^2}{\sigma_s^2 + \sigma_c^2}.
\end{align}
Here, $\mu_s$ and $\mu_c$ correspond to the mean computed across the $n$ dimension for each class. Similarly, $\sigma^2_s$ and $\sigma^2_c$ correspond to the variance computed across the sample dimension $n$ for each class.
Let \(\mathbf{V} = [v_{i,j}] \in \mathbb{R}^{L \times N}\) be the matrix of LDR values, where \(v_{i,j}\) represents the LDR value for the \(i\)-th layer and \(j\)-th token, with \(l\) layers and \(N\) tokens. The LDR output is scaled between 0 and 1 using min-max scaling to allow for consistent comparison across layers.

By visualizing the scaled LDA values as shown in Figure~\ref{fig:concept_representation_methods}, one can identify which layers and tokens exhibit the highest degree of separation between classes, offering insights into the network's internal representations for concept intervention techniques.

\subsection{Concept-Informed
Predictions via Model Steering}\label{sec:steer_concepts}

Through our experiments, we aim to answer the following research questions: \textit{(RQ6)} Can we leverage these learned concepts to guide model output toward concept-informed predictions? For example, can we add periodicity or an upward trend to a constant time series? \textit{(RQ7)} Is it possible to combine multiple steering interventions to manipulate model predictions towards complex compositions of various concepts? For instance, can we steer a model to add both trend and periodicity to a constant signal?

\paragraph{Deriving Steering Matrices for Model Steering.} Once we have identified that a feature is linearly represented in the latent space of the $\mathcal{M}$, we can use steering interventions to manipulate the latent space and generate time series that reflect intended concepts. For instance, to introduce periodicity to a constant time series, we can utilize a steering matrix $\mathbf{S}$, as illustrated in Figure~\ref{fig:concept_representation_methods}. By strategically intervening in $\mathcal{M}$ using this steering matrix, we can bias its outputs towards predicting periodic time series. To construct a steering matrix, we first derive steering vectors $\mathbf{S}_i \in \mathbb{R}^{N \times D}$, for each layer $i$. These vectors represent the change that activations in layer $i$ must undergo such that $\mathcal{M}$ produces periodic outputs. $\mathbf{S}_i$ is simply the difference between the \textit{median} activation matrix of the constant time series $\mathbf{M}_{i_c}$, from that of sinusoids $\mathbf{M}_{i_s}$. We stack these vectors for all layers to derive the steering matrix. This matrix allows us to simultaneously intervene across multiple tokens and layers during inference, which we found to be more effective than single-token interventions. During inference, to steer model predictions, at each layer $i$, we update its hidden representation as follows: $\mathbf{h}_i \leftarrow \mathbf{h}_i + \lambda \mathbf{S}_i,$ where $\lambda \in \mathbb{R}$ is a scalar that controls the strength of the intervention.

We explore two alternative modalities of intervention: (1) deriving steering vectors using the mean of hidden activations rather than the median, and (2) steering a single token versus all tokens throughout the model. While our methods are applicable to a wide range of transformer-based foundation models, we focus on two prominent TSFM families for brevity: \texttt{MOMENT}~\citep{goswami2024moment} and \texttt{Chronos}~\citep{ansari2024chronos}. Both these models are fully open-source, come in different sizes, yet have fundamentally different design choices. For example, \texttt{Chronos} is based on encoder-decoder transformer (in our work we will investigate encoder part) model which takes discretized time series as input, whereas \texttt{MOMENT} is a multi-task, encoder-only model which takes continuous time series patches as input. Since only the \texttt{Large} variant of \texttt{MOMENT} is publicly available at the time of writing this paper, we supplement our representation analysis results with \texttt{Moirai}~\citep{woo2024unified}, another popular TSFM which comes in different sizes. More information on hyper-parameters can be found in Appendix~\ref{app:model_specs}. 

%\footnote{\url{https://github.com/SalesforceAIResearch/uni2ts}}
%\footnote{\url{https://github.com/amazon-science/chronos-forecasting}}
%\footnote{\url{https://github.com/moment-timeseries-foundation-model/moment}}

\section{Results}\label{sec:results}

\paragraph{Analyzing representations offers interesting insights.} Our analysis of model representations demonstrates that both model size and internal architecture considerably influence how representations are organized. Fig.~\ref{fig:separability_heatmaps} shows heatmaps which reveal that larger models, such as \texttt{MOMENT-Large}, \texttt{Chronos-Large}, and \texttt{Moirai-1.1-R Large}, have similar representations across specific groups of layers forming distinct and intricate block patterns, which may reflect unique stages of representation learning. More complex block patterns are observed with increasing model size, indicating that scaling may enhance the richness and organization of internal representations. However, it may also increase redundant knowledge storage through similar representations across layers, as suggested by high CKA similarity measured in block patterns. Interestingly, within model families (e.g., \texttt{Chronos} and \texttt{Moirai}), scaling does not always result in predictable heatmap changes. Larger models, like \texttt{Chronos-Large} and \texttt{Moirai-Large}, demonstrate more refined and complex transformations of representations that are not easily extrapolated from their smaller versions as shown in Fig.~\ref{fig:model_representation_scaling}). Moreover, cross-model similarity analysis results in Fig.~\ref{fig:model_similarity_across_family} reveal that while early layers tend to have high similarity across models of different sizes, the similarity measures among later layers diverge more notably, particularly in larger models. This divergence is especially evident in the \texttt{Chronos} family, where early representations are more consistent across models, but later layers become increasingly specialized as model depth increases as shown in Fig.~\ref{fig:model_family_self_similarity}.



% \begin{figure}[ht!]
% \centering
% \setlength{\tabcolsep}{0pt} 
% \begin{tabular}{cc}
% \multicolumn{2}{c}{\texttt{Chronos Family}} \\
% \includegraphics[height=2.5cm, trim={4.0cm 11cm 4cm 4.5cm}, clip]{iclr2025/figures/single_family_intermodel_chronos/tiny_vs_mini.pdf} & \includegraphics[height=2.5cm, trim={3cm 11cm 4cm 14.5cm}, clip]{iclr2025/figures/single_family_intermodel_chronos/base_vs_large.pdf} \\  
% (i) \texttt{Tiny vs Mini} &
% (ii) \texttt{Base vs Large} \\ 
% \end{tabular}
% \caption{Similarity between representations learned by different layers in TSFMs of the same family but different sizes. Initial layers tend to learn similar representations, while the similarity gradually decreases in the later layers.}
% \label{fig:model_similarity_across_family}
% \end{figure}

% \begin{figure}[ht!]
% \centering
% \setlength{\tabcolsep}{0pt} % Default 
% \begin{tabular}{cccc}
% \includegraphics[width=0.16\textwidth, trim={3.0cm 11cm 4cm 4.5cm}, clip]{iclr2025/figures/large_redundant/moment.pdf} &  
% \includegraphics[width=0.16\textwidth, trim={3.0cm 11cm 4cm 4.5cm}, clip]{iclr2025/figures/large_redundant/chronos.pdf} &
% \includegraphics[width=0.16\textwidth, trim={3.0cm 11cm 4cm 4.5cm}, clip]{iclr2025/figures/large_redundant/moirai-1.1.pdf}  \\
% (i) \texttt{MOMENT} & (ii) \texttt{Chronos} & (iii) \texttt{Moirai} \\ 
% \end{tabular}
% \caption{Pairwise similarity of layer measured using CKA for large variants of TSFMs (\textcolor{darkblue}{dark blue} $\rightarrow$ low similarity, \textcolor{yellow}{yellow} $\rightarrow$ high similarity). All TSFMs learn redundant representations with manifesting as block-like structures.}
% \label{fig:large_models_self_similarity}
% \end{figure}

\paragraph{Block-wise pruning can improve model throughput, without compromising accuracy.} We observed consistent improvements in memory efficiency and inference speed over their unpruned counterparts. For example, pruning only Block 3 for \texttt{MOMENT-Large}, resulted in a 11\% decrease in estimated model size with a 5\% speed up in inference. Furthermore, this pruned model had lower zero-shot imputation MAE for 5 of 9 datasets (ETTh2, ETTm1, ETTm2, Exchange, and Weather) as shown in Tab.~\ref{tab:reduced_zero_hot_imputation}. \texttt{Chronos-Large} results for zero-shot experiments are reported in Tab. \ref{tab:chronos_zero_shot}. Detailed results on memory usage and speed improvements can be found in Tab. \ref{tab:inference_performance}. While pruning consistently improved memory efficiency and inference speed compared to unpruned counterparts, performance varied across pruning methods and datasets, with some methods exhibiting considerable degradation. In addition to zero-shot experiments, we conducted experiments where models were fine-tuned post-pruning. For this, we applied \texttt{MOMENT-Large} for forecasting to compare a vanilla (unpruned) model to one with all block redundancies pruned, evaluating the impact of the most aggressive pruning approach. Fine-tuning results in Table~\ref{tab:finetuning_forecasting_main} show that, notably, the pruned model performed nearly as well as the original, underscoring the potential of our block-wise pruning approach to maintain performance while reducing model complexity. Complete finetuning results are provided in Table~\ref{tab:finetuning_forecasting_apd} in Appendix~\ref{app:additional_results}.


\begin{figure}[t!]
\centering
\setlength{\tabcolsep}{0pt} 
\begin{tabular}{cc}
\multicolumn{2}{c}{\texttt{Chronos Family}} \\
\includegraphics[height=2.5cm, trim={4.0cm 11cm 4cm 4.5cm}, clip]{iclr2025/figures/single_family_intermodel_chronos/tiny_vs_mini.pdf} & \includegraphics[height=2.5cm, trim={3cm 11cm 4cm 14.5cm}, clip]{iclr2025/figures/single_family_intermodel_chronos/base_vs_large.pdf} \\  
(i) \texttt{Tiny vs Mini} &
(ii) \texttt{Base vs Large} \\ 
\end{tabular}
\caption{Similarity between representations learned by different layers in TSFMs of the same family but different sizes. Initial layers tend to learn similar representations, while the similarity gradually decreases in the later layers.}
\label{fig:model_similarity_across_family}
\end{figure}

\begin{figure}[t!]
\centering
\setlength{\tabcolsep}{0pt} % Default 
\begin{tabular}{cccc}
\includegraphics[width=0.16\textwidth, trim={3.0cm 11cm 4cm 4.5cm}, clip]{iclr2025/figures/large_redundant/moment.pdf} &  
\includegraphics[width=0.16\textwidth, trim={3.0cm 11cm 4cm 4.5cm}, clip]{iclr2025/figures/large_redundant/chronos.pdf} &
\includegraphics[width=0.16\textwidth, trim={3.0cm 11cm 4cm 4.5cm}, clip]{iclr2025/figures/large_redundant/moirai-1.1.pdf}  \\
(i) \texttt{MOMENT} & (ii) \texttt{Chronos} & (iii) \texttt{Moirai} \\ 
\end{tabular}
\caption{Pairwise similarity of layer measured using CKA for large variants of TSFMs (\textcolor{darkblue}{dark blue} $\rightarrow$ low similarity, \textcolor{yellow}{yellow} $\rightarrow$ high similarity). All TSFMs learn redundant representations with manifesting as block-like structures.}
\label{fig:large_models_self_similarity}
\end{figure}



\begin{table}[!htb]
\centering
\resizebox{0.47\textwidth}{!}{
\begin{tabular}{c|cccccc}
\toprule
 & ETTh1 & ETTh2 & ETTm1 & ETTm2 & ILI & Weather \\ \midrule
Vanilla & \textbf{0.385} & \textbf{0.287} & \textbf{0.290} & \textbf{0.171} & 3.260 & 0.153 \\
Pruned & 0.388 & 0.296 & \textbf{0.290} & 0.173 & \textbf{2.981} & \textbf{0.152} \\
\bottomrule
\end{tabular}}
\caption{MSE of fine-tuned vanilla and pruned \texttt{MOMENT} variants on long-horizon forecasting datasets~\citep{Informer}. Pruned models perform on par with the original, whilereducing memory consumption by $>50\%$ and improving inference time per sample by $\approx$1 ms. More comprehensive results are available in the appendix in Tables \ref{tab:finetuning_forecasting_apd} and \ref{tab:inference_performance}.
}
% ref tables 6 and 7
\label{tab:finetuning_forecasting_main}
\end{table}



\begin{figure}[ht!]
\centering
\setlength{\tabcolsep}{0pt} % Default 
\begin{tabular}{ccccc}
\multicolumn{5}{c}{\texttt{Chronos Family}} \\
\includegraphics[width=0.095\textwidth, trim={4.5cm 12cm 4cm 4.5cm}, clip]{iclr2025/figures/scaling_family_chronos/tiny.pdf} &  
\includegraphics[width=0.095\textwidth, trim={4.5cm 12cm 4cm 4.5cm}, clip]{iclr2025/figures/scaling_family_chronos/mini.pdf} &
\includegraphics[width=0.095\textwidth, trim={4.5cm 12cm 4cm 4.5cm}, clip]{iclr2025/figures/scaling_family_chronos/small.pdf} &
\includegraphics[width=0.095\textwidth, trim={4.5cm 12cm 4cm 4.5cm}, clip]{iclr2025/figures/scaling_family_chronos/base.pdf} &
\includegraphics[width=0.095\textwidth, trim={4.5cm 12cm 4cm 4.5cm}, clip]{iclr2025/figures/scaling_family_chronos/large.pdf}  \\
(i) \texttt{Tiny} & (ii) \texttt{Mini} & (iii) \texttt{Small} & (iv) \texttt{Base} & (v) \texttt{Large} \\ 
\end{tabular}
\caption{How does model size influence the patterns of learned representations? The emergence of blocks-like patterns in the \texttt{Large} model appears unpredictable from patterns observed in smaller models. More available in the appendix (Fig. \ref{fig:model_family_self_similarity_apd}).}
\label{fig:model_family_self_similarity}
\end{figure}

\begin{figure}[ht!]
\centering
\setlength{\tabcolsep}{2pt}
\begin{tabular}{ccc}
\includegraphics[width=0.15\textwidth, trim={2.7cm 2.7cm 9cm 1.1cm}, clip]{iclr2025/figures/linear_separability_heatmaps/pattern.pdf} &  
\includegraphics[width=0.15\textwidth, trim={2.7cm 2.7cm 9cm 1.1cm}, clip]{iclr2025/figures/linear_separability_heatmaps/trend.pdf} &
\includegraphics[width=0.15\textwidth, trim={2.7cm 2.7cm 9cm 1.1cm}, clip]{iclr2025/figures/linear_separability_heatmaps/periodicity.pdf} \\
% \includegraphics[width=0.16\textwidth, trim={0.0cm 0cm 4cm 1.1cm}, clip]{iclr2025/figures/linear_separability_heatmaps/amplitude.pdf} \\
(i) Pattern & (ii) Trend & (iii) Periodicity\\ 
\end{tabular}
\caption{Progression of linear separability of concepts at the patch level (y-axis) across different layers (x-axis). Linear separability is measured using the Linear Discriminant Ratio (LDR), derived from model embedding statistics for each predicted class: (i) constant vs. sinusoidal patterns, (ii) increasing vs. decreasing trends, and (iii) high vs. low periodicity. Color gradient indicates separability, with \textcolor{darkblue}{dark blue} representing low LDR and \textcolor{yellow}{yellow} indicating high LDR. \textit{Certain concepts captured by MOMENT-Large exhibit linear separability, but this separability is not uniform across layers; instead, it emerges at specific points in the model.} More available in the appendix (Fig. \ref{fig:separability_heatmaps_apd}).
}
\label{fig:separability_heatmaps}
\end{figure}



\paragraph{TSFMs learn intuitive linear concepts.} Our concept localization results in Fig.~\ref{fig:separability_heatmaps} show that certain concepts represented by \texttt{MOMENT-Large} are linearly separable and that this separability is not consistent but rather emerges at specific layers in the model. We also found intuitive differences in the locations where these concepts are learned. We observed that certain concepts, such as distinguishing between constant and sinusoidal patterns, require examination of the entire time series. In contrast, differentiating between increasing and decreasing trends can be achieved by focusing on the initial and final patches. However, we did not identify specific locations where models learn to distinguish between time series of different amplitudes. This may be attributed to the normalization of input time series, a common practice in many TSFMs, including \texttt{MOMENT}.


\begin{figure*}[!t]
\centering
\setlength{\tabcolsep}{0pt} % Default value: 6pt
\begin{tabular}{ccc}
\toprule
\multicolumn{3}{c}{\textbf{Steering:} \textit{Introduce periodicity (i) and trend (ii, iii) to constant time series}} \\ 
\includegraphics[width=0.3\textwidth, trim={0cm 0cm 0cm 0cm}, clip]{iclr2025/figures/output_space_steering/constant_to_sine.pdf} &  
\includegraphics[width=0.3\textwidth, trim={0cm 0cm 0cm 0cm}, clip]{iclr2025/figures/output_space_steering/constant_to_increasing.pdf} &  
\includegraphics[width=0.3\textwidth, trim={0cm 0cm 0cm 0cm}, clip]{iclr2025/figures/output_space_steering/constant_to_decreasing.pdf} \\ 
(i) Periodic & (ii) Increasing Trend & (iii) Decreasing Trend \\ \midrule
\multicolumn{3}{c}{\textbf{Compositional Steering:} \textit{Introduce trend and periodicity to constant time series}} \\ 
\multicolumn{3}{c}{(\texttt{MOMENT} (top), \texttt{Chronos} (bottom))} \\ 
\includegraphics[width=0.3\textwidth, trim={0cm 0cm 0cm 0cm}, clip]{iclr2025/figures/compositional_steering_MOMENT/alpha_0.0.pdf} &
\includegraphics[width=0.3\textwidth, trim={0cm 0cm 0cm 0cm}, clip]{iclr2025/figures/compositional_steering_MOMENT/alpha_0.5.pdf} &
\includegraphics[width=0.3\textwidth, trim={0cm 0cm 0cm 0cm}, clip]{iclr2025/figures/compositional_steering_MOMENT/alpha_1.0.pdf} \\
\includegraphics[width=0.3\textwidth, trim={0cm 0cm 0cm 0cm}, clip]{iclr2025/figures/compositional_steering_CHRONOS_appendix/alpha_1.0.pdf} &
\includegraphics[width=0.3\textwidth, trim={0cm 0cm 0cm 0cm}, clip]{iclr2025/figures/compositional_steering_CHRONOS_appendix/alpha_0.5.pdf} &
\includegraphics[width=0.3\textwidth, trim={0cm 0cm 0cm 0cm}, clip]{iclr2025/figures/compositional_steering_CHRONOS_appendix/alpha_0.0.pdf} \\
(i) Trend, $\beta = 1.0$ & (ii) Trend \& periodicity, $\beta = 0.5$ & (iii) Periodicity, $\beta = 0$ \\ 
\bottomrule
\end{tabular}
\caption{\textbf{Visualization of \texttt{MOMENT} reconstruction and \texttt{Chronos} forecasting predictions (bottom), with and without concept steering applied in the latent space.} To evaluate the effects of concept steering, we provide a constant time series as input and visualize model predictions with (blue) and without (orange) steering applied. The original constant signal and steered output are referred to as \textit{Non-perturbed} and \textit{Perturbed}, respectively. The non-perturbed output remains a constant signal as expected, while the perturbed output reflects the new concept introduced via the steering matrix—such as trend, seasonality, or both—depending on the $\beta$ parameter. For both outputs, we show the raw model predictions (lighter color) and their moving averages (darker color) to reduce noise artifacts. Steering results are shown for the following experiments: (i) steering a constant signal to produce a sinusoidal output, (ii) producing a constant signal with an increasing trend (slope $> 0$), and (iii) producing a decreasing trend (slope $< 0$). In compositional steering experiments, the parameter $\beta$ controls the blend of sinusoidal and increasing trend concepts. \textit{When $\beta = 0.5$, the model is steered toward a combination of an increasing trend and a sinusoidal pattern.} Setting $\beta = 0$ steers the model toward an increasing trend, while $\beta = 1$ introduces only the sinusoidal component (iii). Detailed results are available in Appendix~\ref{app:additional_results}.
}
\label{fig:moment_output_space}
\end{figure*}

\paragraph{We can effectively steer TSFM predictions.} Our concept steering interventions effectively transform the latent space of TSFMs, resulting in model predictions that align with the intended concepts, as demonstrated in Fig.~\ref{fig:moment_output_space}. We successfully introduced periodicity and trend concepts to constant time series and demonstrated the ability to combine multiple steering vectors to create more complex patterns. By combining steering vectors representing increasing trends and sinusoidal patterns, we were able to steer model predictions towards a combination of these features. To assess the robustness of concept steering, we generated datasets using multiple random seeds and confirmed that consistent steering effects emerge across samples. This is supported by analyses of linear concept separability and concept emergence in the latent space. Furthermore, in Appendix~\ref{app:additional_results}, we present steering results on real-world ECG data, where we steer time series from the normal to abnormal heartbeat class in the ECG5000 dataset. The ability of our proposed concept steering method to generalize across both synthetic and real-world data highlights its robustness and effectiveness.

To assess the impact of steering in the latent space, we analyzed changes in the hidden representations before and after applying concept steering by projecting them into a two-dimensional space using Principal Component Analysis (PCA). We found that steering in the latent space is reflected in these lower-dimensional representations, as illustrated in Fig.~\ref{fig:pca_steering}. Notably, the PCA reduction often captured the concept direction as one of the principal components. This can be attributed to the careful design of our synthetic data generation process.

Interestingly, the method of obtaining the steering matrix, either by computing the mean or median across embedding concept classes, has no notable effect on the steered output as shown in Fig.~\ref{fig:intervention_comparison}. However, applying concept steering interventions across all tokens is necessary to achieve the intended steered concept output compared to applying concept steering interventions to a single token. Moreover, the $\lambda$ parameter can have considerable effect on steered output. For \texttt{Chronos}, steering required tuning the parameter $\lambda\approx 0.1$ for effective performance, whereas \texttt{MOMENT} maintained effective steering with $\lambda = 1$.