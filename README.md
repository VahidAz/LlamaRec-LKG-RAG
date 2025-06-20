# LlamaRec-LKG-RAG

This repository is the impelementation for the <font size='5'>**LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking**</font> <a href='https://www.arxiv.org/abs/2506.07449'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

<p align="center">
      <img src=model.png width=800, height=500>
</p>

Recent advances in Large Language Models (LLMs) have driven their adoption in recommender systems through Retrieval-Augmented Generation (RAG) frameworks. However, existing RAG approaches predominantly rely on flat, similarity-based retrieval that fails to leverage the rich relational structure inherent in user-item interactions. We introduce LlamaRec-LKG-RAG, a novel single-pass, end-to-end trainable framework that integrates personalized knowledge graph context into LLM-based recommendation ranking. Our approach extends the LlamaRec architecture by incorporating a lightweight user preference module that dynamically identifies salient relation paths within a heterogeneous knowledge graph constructed from user behavior and item metadata. These personalized subgraphs are seamlessly integrated into prompts for a fine-tuned Llama-2 model, enabling efficient and interpretable recommendations through a unified inference step. Comprehensive experiments on ML-100K and Amazon Beauty datasets demonstrate consistent and significant improvements over LlamaRec across key ranking metrics (MRR, NDCG, Recall). LlamaRec-LKG-RAG demonstrates the critical value of structured reasoning in LLM-based recommendations and establishes a foundation for scalable, knowledge-aware personalization in next-generation recommender systems.

## Setup Instructions

### 1. Creating the Conda Environment
To set up the required environment, run the following command:
```sh
conda env create -f environment.yml
```
Then
```sh
conda activate llamarec-lkg-rag
```

### 2. Training the Retriever Model
Navigate to the `LlamaRec` directory and run the training script for retriever:
```sh
cd LlamaRec
python train_retriever.py
```
Run this script for both the `ml-100k` and `beauty` datasets. Enter 1 for ml-100k and b for beauty.
For more details, check the [LlamaRec repository](https://github.com/Yueeeeeeee/LlamaRec).

### 3. Neo4j Graph Creation Guide
Before you begin, you need to familiarize yourself with Neo4j. You can explore Neo4j on their [official website](https://neo4j.com).  
You also need to create an account on Neo4j Aura. Details on how to create an account can be found in the [Neo4j Aura Account Setup](https://neo4j.com/docs/aura/classic/platform/create-account/) documentation.

## MovieLens Dataset
1. Extract the metadata from IMDb by `extract_movielens_metadata.ipynb`. The metadata is already available in `ml-latest-small/movies_metadata.csv`.
2. Create an AuraDB Free instance, as the graph for the ML-100K dataset is small.
3. Run the `create_graph_movielens.ipynb` notebook to create the graph. Please ensure that your Neo4j credentials are set in the notebook. A snapshot of the graph is available as neo4j-beauty.backup, which you can restore instead of recreating the graph (restoring is faster). For instructions on how to restore a backup file in Neo4j, refer to the [Backup and Restore Documentation](https://neo4j.com/docs/aura/managing-instances/backup-restore-export/).

## Beauty Dataset
1. Create an AuraDB Professional instance, as the graph for the Beauty dataset is large. There is a 14-day trial for AuraDB Professional, which is extendable to 21 days.
2. Run the `create_graph_beauty.ipynb` notebook to create the graph. Please ensure that your Neo4j credentials are set in the notebook. A snapshot of the graph is available as neo4j-beauty.backup, which you can restore instead of recreating the graph (restoring is faster). For instructions on how to restore a backup file in Neo4j, refer to the [Backup and Restore Documentation](https://neo4j.com/docs/aura/managing-instances/backup-restore-export/).

### 3. Training


## Acknowledgement
+ [LlamaRec](https://github.com/Yueeeeeeee/LlamaRec) This repository is built upon LlamaRec.

## Citation
Please consider citing the following paper if you use our method in your research:
```bibtex
@misc{azizi2025llamareclkgragsinglepasslearnableknowledge,
      title={LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking}, 
      author={Vahid Azizi and Fatemeh Koochaki},
      year={2025},
      eprint={2506.07449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2506.07449}, 
}
```
