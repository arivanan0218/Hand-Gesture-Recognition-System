# Hand Gesture Recognition

This project is a Hand Gesture Recognition system using a Convolutional Neural Network (CNN) built with Keras and TensorFlow. The model is trained to classify six hand gestures: NONE, ONE, TWO, THREE, FOUR, and FIVE.

## Requirements
- TensorFlow
- Keras
- NumPy
- scikit-learn

Install dependencies with:
```
pip install tensorflow keras scikit-learn
```

## Training the Model
To train the model, run the following command:
```
python Train.py
```

### Key Features in Training:
- CNN architecture with Conv2D, BatchNormalization, and Dropout layers.
- Data augmentation for improved model generalization.
- Class weights to balance imbalanced datasets.
- Early stopping and model checkpoint callbacks for efficient training.

## Testing the Model
To classify test images, run the following command:
```
python Test.py
```

### Sample Output:
```
Loaded model from disk
ARRAY [0.05 0.10 0.20 0.25 0.15 0.25]
Img_name HandGestureDataset/train/TWO/img001.png RESULT TWO
```

## Model Saving and Loading
- The trained model's architecture is saved in `model.json`.
- Model weights are saved in `model.weights.h5`.

## Future Improvement
- Integrate with a real-time webcam interface for live recognition.


