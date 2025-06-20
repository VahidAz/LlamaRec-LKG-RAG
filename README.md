# LlamaRec-

## Setup Instructions

### 1. Creating the Conda Environment
To set up the required environment, run the following command:
```sh
conda env create -f environment.yml
```
Then
```sh
conda activate llamarec-lgrag
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

## Acknowledgement
+ [LlamaRec](https://github.com/Yueeeeeeee/LlamaRec) This repository is built upon LlamaRec.


## License
This repository is licensed under the [BSD 3-Clause License](LICENSE).

