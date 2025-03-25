CREATE TABLE cnn_training_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    epoch INT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    train_loss FLOAT,
    train_accuracy FLOAT,
    val_loss FLOAT,
    val_accuracy FLOAT,
    early_stop BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);