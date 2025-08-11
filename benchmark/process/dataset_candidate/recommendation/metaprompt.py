TASK = r"""
Train a recommendation model for the given dataset.
"""


DATASET = r"""
The datasets for recommendation: \textbf{Yelp} (29,601 users, 24,734 items, 1,069,128 interactions): a dataset collected from the rating interactions on Yelp platform; \textbf{Gowalla} (50,821 users, 57,440 items, 1,172,425 interactions): a dataset containing users' check-in records collected from Gowalla platform; \textbf{ML-10M} (69,878 users, 10,195 items, 6,999,171 interactions): a well-known movie-rating dataset for collaborative filtering; \textbf{Amazon-book} (78,578 users, 77,801 items, 2,240,156 interactions): a dataset composed of users' ratings on books collected from Amazon;

The datasets have already been downloaded in the directory `/workplace/dataset_candidate`, move them to `/workplace/project/data`.
"""

BASELINE = r"""
• GNN-based Collaborative Filtering: LightGCN [1].
• Disentangled Graph Collaborative Filtering: DGCF [3].
• Hypergraph-based Collaborative Filtering: HyRec [2].
• Self-Supervised Learning Recommender Systems: GCA [10], MHCN [8], SimGRACE [5], SGL [4], HCCF [6], SHT [7], SimGCL [9]

References
[1] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. LightGCN: Simplifying and powering graph convolution network for recommendation. In International conference on research and development in Information Retrieval (SIGIR), pages 639–648, 2020.
[2] Jianling Wang, Kaize Ding, Liangjie Hong, Huan Liu, and James Caverlee. Next-item recommendation with sequential hypergraphs. In Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval, pages 1101–1110, 2020.
[3] Xiang Wang, Hongye Jin, An Zhang, Xiangnan He, Tong Xu, and Tat-Seng Chua. Disentangled graph collaborative filtering. In Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval, pages 1001–1010, 2020.
[4] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. Self-supervised graph learning for recommendation. In International conference on research and development in information retrieval (SIGIR), pages 726–735, 2021.
[5] Jun Xia, Lirong Wu, Jintao Chen, Bozhen Hu, and Stan Z Li. Simgrace: A simple framework for graph contrastive learning without data augmentation. In the ACM Web Conference (WWW), pages 1070–1079, 2022.
[6] Lianghao Xia, Chao Huang, Yong Xu, Jiashu Zhao, Dawei Yin, and Jimmy Xiangji Huang. Hypergraph contrastive collaborative filtering. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2022.
[7] Lianghao Xia, Chao Huang, and Chuxu Zhang. Self-supervised hypergraph transformer for recommender systems. In International Conference on Knowledge Discovery and Data Mining, KDD 2022, Washington DC, USA, August 14-18, 2022., 2022.
[8] Lianghao Xia, Chao Huang, and Chuxu Zhang. Self-supervised hypergraph transformer for recommender systems. In International Conference on Knowledge Discovery and Data Mining, KDD 2022, Washington DC, USA, August 14-18, 2022., 2022.
[9] Junliang Yu, Hongzhi Yin, Jundong Li, Qinyong Wang, Nguyen Quoc Viet Hung, and Xiangliang Zhang. Self-supervised multi-channel hypergraph convolutional network for social recommendation. In Proceedings of the Web Conference 2021, pages 413–424, 2021.
[10] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Quoc Viet Hung Nguyen. Are graph augmentations necessary? simple graph contrastive learning for recommendation. In International Conference on Research and Development in Information Retrieval (SIGIR), pages 1294–1303, 2022.
"""

COMPARISON = r"""
\begin{table}[h]
\setlength{\tabcolsep}{3pt}
\centering
\caption{Performance comparison with baselines on 4 datasets.}
\label{mainresult}
\scriptsize
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
Data & Metric & DGCF & HyRec & {\tiny LightGCN} &  MHCN & SGL & {\tiny SimGRACE} & GCA & HCCF & SHT & SimGCL & LightGCL \\

\hline
\multirow{4}{*}{\rotatebox{90}{Yelp}}
&R@20 & 0.0466 & 0.0472 & 0.0482 &  0.0503 & 0.0526 & 0.0603 & 0.0621 & 0.0626 & 0.0651 & 0.0718 & {0.0793}  \\

&N@20 & 0.0395 & 0.0395  & 0.0409 &  0.0424 & 0.0444 & 0.0435 & 0.0530 & 0.0527 & 0.0546 & 0.0615 & {0.0668}  \\
\cline{2-13}

&R@40 & 0.0774 & 0.0791 & 0.0803 &  0.0826 & 0.0869 & 0.0989 & 0.1021 & 0.1040 & 0.1091 & 0.1166 & {0.1292}  \\

&N@40 & 0.0511 & 0.0522 & 0.0527 &  0.0544 & 0.0571 & 0.0656 & 0.0677 & 0.0681 & 0.0709 & 0.0778 & {0.0852}  \\
\hhline{|=|=|=|=|=|=|=|=|=|=|=|=|=|}
\multirow{4}{*}{\rotatebox{90}{Gowalla}}
&R@20 & 0.0944 & 0.0901 & 0.0985 &  0.0955 & 0.1030 & 0.0869 & 0.0896 & 0.1070 & 0.1232 & 0.1357 & {0.1578}  \\

&N@20 & 0.0522 & 0.0498 & 0.0593 &  0.0574 & 0.0623 & 0.0528 & 0.0537 & 0.0644 & 0.0731 & 0.0818 & {0.0935}  \\
\cline{2-13}

&R@40 & 0.1401 & 0.1356 & 0.1431 &  0.1393 & 0.1500 & 0.1276 & 0.1322 & 0.1535 & 0.1804 & 0.1956 & {0.2245}  \\

&N@40 & 0.0671 & 0.0660 & 0.0710 &  0.0689 & 0.0746 & 0.0637 & 0.0651 & 0.0767 & 0.0881 & 0.0975 & {0.1108}  \\
\hhline{|=|=|=|=|=|=|=|=|=|=|=|=|=|}
\multirow{4}{*}{\rotatebox{90}{ML-10M}}
&R@20 & 0.1763 & 0.1801 & 0.1789 &  0.1497 & 0.1833 & 0.2254 & 0.2145 & 0.2219 & 0.2173 & 0.2265 & {0.2613}  \\

&N@20 & 0.2101 & 0.2178 & 0.2128 &  0.1814 & 0.2205 & 0.2686 & 0.2613 & 0.2629 & 0.2573 & 0.2613 & {0.3106}  \\
\cline{2-13}

&R@40 & 0.2681 & 0.2685 & 0.2650 &  0.2250 & 0.2768 & 0.3295 & 0.3231 & 0.3265 & 0.3211 & 0.3345 & {0.3799}  \\

&N@40 & 0.2340 & 0.2340 & 0.2322 &  0.1962 & 0.2426 & 0.2939 & 0.2871 & 0.2880 & 0.3318 & 0.2880 & {0.3387}  \\
\hhline{|=|=|=|=|=|=|=|=|=|=|=|=|=|}
\multirow{4}{*}{\rotatebox{90}{Amazon}}
&R@20 & 0.0211 & 0.0302 & 0.0319 &  0.0296 & 0.0327 & 0.0381 & 0.0309 & 0.0322 & 0.0441 & 0.0474 & {0.0585}  \\

&N@20 & 0.0154 & 0.0225 & 0.0236 &  0.0219 & 0.0249 & 0.0291 & 0.0238 & 0.0247 & 0.0328 & 0.0360 & {0.0436}  \\

\cline{2-13}
&R@40 & 0.0351 & 0.0432 & 0.0499 &  0.0489 & 0.0531 & 0.0621 & 0.0498 & 0.0525 & 0.0719 & 0.0750 & {0.0933}   \\

&N@40 & 0.0201 & 0.0246 & 0.0290 &  0.0284 & 0.0312 & 0.0371 & 0.0301 & 0.0314 & 0.0420 & 0.0451 & {0.0551}   \\
\hline
\end{tabular}
\label{tab:overall_performance}
\end{table}
"""

EVALUATION = r"""
Adopt the Recall@N and Normalized Discounted Cumulative Gain (NDCG)@N, where N = \{20, 40\}, as the evaluation metrics.
The code of evaluation metrics could be:

\begin{verbatim}
import numpy as np

def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num
\end{verbatim}
"""

REF = r"""
All this information is from LightGCL paper, and the repository of LightGCL is in the directory `/workplace/dataset_candidate/LightGCL`, you can refer to it when you need to process the dataset or calculate the metrics. 

[IMPORTANT]
1. You should train the model on the train set for several epochs, and then evaluate the model on the test set only once. DO NOT evaluate the model EVERY EPOCH.
2. The training process should follow a mini-batch training strategy, each batch contains multiple users (e.g., batch_size=2048 users). You can refer to LightGCL for more details.
"""