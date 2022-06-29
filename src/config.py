config = {
    "src_train": "../data-set/train.de",
    "trg_train":  "../data-set/train.en",
    "src_valid": "../data-set/val.de",
    "trg_valid":  "../data-set/val.en",
    "HID_DIM": 256,
    "ENC_LAYERS": 3,
    "DEC_LAYERS": 3,
    "ENC_HEADS": 8,
    "DEC_HEADS": 8,
    "ENC_PF_DIM": 512,
    "DEC_PF_DIM": 512,
    "ENC_DROPOUT": 0.1,
    "DEC_DROPOUT":  0.1,
    "N_EPOCHS":  10,
    "CLIP": 1,
    "learning_rate": 0.0005,
    "test_config": {
        "model_path": "model.pt",
        "src_test": "../data-set/test_2016_flickr.de",
        "trg_test": "../data-set/test_2016_flickr.en",
    }
}
