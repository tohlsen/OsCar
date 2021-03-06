{
    "dataset_reader": {
        "type": "drop",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-cased",
                "do_lowercase": true,
                "use_starting_offsets": false,
            },
        },
        "passage_length_limit": 200,
        "question_length_limit": 50,
        "skip_when_all_empty": ["passage_span", "question_span", "addition_subtraction", "counting"],
        "instance_format": "drop"
    },
    "validation_dataset_reader": {
        "type": "drop",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-cased",
                "do_lowercase": true,
                "use_starting_offsets": false,
            },
        },
        "passage_length_limit": 500,
        "question_length_limit": 100,
        "skip_when_all_empty": [],
        "instance_format": "drop"
    },



    "train_data_path": "drop_dataset/drop_dataset_mini.json",
    "validation_data_path": "drop_dataset/drop_dataset_dev.json",
    
    "model": {
        "type": "naqanet",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"],
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "bert-base"
                },
            }
        },
        "num_highway_layers": 2,
        "phrase_layer": {
            "type": "qanet_encoder",
            "input_dim": 128,
            "hidden_dim": 128,
            "attention_projection_dim": 128,
            "feedforward_hidden_dim": 128,
            "num_blocks": 1,
            "num_convs_per_block": 4,
            "conv_kernel_size": 7,
            "num_attention_heads": 8,
            "dropout_prob": 0.1,
            "layer_dropout_undecayed_prob": 0.1,
            "attention_dropout_prob": 0
        },
        "matrix_attention_layer": {
            "type": "linear",
            "tensor_1_dim": 128,
            "tensor_2_dim": 128,
            "combination": "x,y,x*y"
        },
        "modeling_layer": {
            "type": "qanet_encoder",
            "input_dim": 128,
            "hidden_dim": 128,
            "attention_projection_dim": 128,
            "feedforward_hidden_dim": 128,
            "num_blocks": 1,
            "num_convs_per_block": 2,
            "conv_kernel_size": 5,
            "num_attention_heads": 8,
            "dropout_prob": 0.1,
            "layer_dropout_undecayed_prob": 0.1,
            "attention_dropout_prob": 0
        },
        "dropout_prob": 0.1,
        "regularizer": [
            [
                ".*",
                {
                    "type": "l2",
                    "alpha": 1e-07
                }
            ]
        ],
        "answering_abilities": [
            "passage_span_extraction",
            "question_span_extraction",
            "addition_subtraction",
            "counting"
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "passage",
                "num_tokens"
            ],
            [
                "question",
                "num_tokens"
            ]
        ],
        "batch_size": 16,
        "max_instances_in_memory": 600
    },
    "trainer": {
        "num_epochs": 5,
        "grad_norm": 5,
        "patience": 10,
        "num_serialized_models_to_keep": 1,
        "validation_metric": "+f1",
        "cuda_device": 1,
        "optimizer": {
            "type": "adam",
            "lr": 5e-4,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-07
        },
        "moving_average": {
            "type": "exponential",
            "decay": 0.9999
        }
    }
}
