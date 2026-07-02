To download the model and precomputed embeddings: https://drive.google.com/drive/folders/1jr0J7fizeugKoOvDG-sSCTFXzIQc-xJY?usp=drive_link 

The downloaded files from drive need to be in their respective paths as shown in the project structure below:
  - emb_shard_{N}.pt in /output/embeddings
  - best_model.pt in /output/checkpoints

Read /output/logs/commands_used.txt for a guide on how to use Penguin.


```/
├── dataset
│   ├── data
│   │   ├── location_xxxxxx.json
│   │   ├── location_xxxxxx.png
│   │   └── ...
│   └── data_mapped
│   │   ├── location_xxxxxx.json
│   │   ├── location_xxxxxx.png
│   │   └── ...
├── output
│   ├── centroids
│   │   └── centroids_ab85c9672ec4.pt
│   ├── checkpoints
│   │   └── best_model.pt
│   ├── embeddings
│   │   ├── country_texts
│   │   │   ├── country_embeddings.pt
│   │   │   └── country_prompts.json
│   │   ├── embedding_index.json
│   │   ├── emb_shard_0000.pt
│   │   ├── ...
│   │   └── emb_shard_0033.pt
│   ├── feature_test
│   │   ├── image_aug0_boxes.png
│   │   ├── image_aug0.png
│   │   ├── image_aug1_boxes.png
│   │   ├── image_aug1.png
│   │   └── image_boxes.png
│   ├── features
│   │   ├── feat_shard_0000.pt
│   │   ├── ...
│   │   ├── feat_shard_0033.pt
│   │   └── feature_index.json
│   ├── logs
│   │   ├── commands_used.txt
│   │   └── training_log.txt
│   └── play
│       ├── rounds
│       │   ├── round{N}
│       │   │   ├── result.png
│       │   │   └── streetview.png
│       │   └── ...
│       └── runs
│           ├── run_timestamp.json
│           └── ...
├── plonkit_data
│   ├── countries
│   │   ├── country_name.json
│   │   └── ...
│   └── plonkit_db.json
├── road_model
│   └── yolov8m-worldv2.pt
├── weights # Yolo_world dependency
│   └── clip
│       └── ViT-B-32.pt
├── config.py
├── dataset.py
├── eval.py
├── features.py
├── model.py
├── play_openguessr.py
├── plonkit_integration.py
├── plonkit.py
├── precompute.py
├── test_augment.py
├── test_feature_detection.py
├── train.py
├── utils.py
└── requirements.txt
```
