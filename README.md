Emotion Detection Project Summary

This document provides an overview of the iterative process and findings during the development of an emotion detection model as part of my academic project. The journey from initial attempts to the final approach highlights the challenges and learning experiences encountered along the way.
Initial Exploration

    First Attempt: Launched the model training for the first time with 10 epochs, achieving an 83% accuracy on the training set but only 51% on the test set, indicating a clear case of overfitting.

Refinements and Adjustments

    Validation Set Introduction: Added a validation set and implemented early stopping to combat overfitting, but saw no significant improvements.
    Data Augmentation: Experimented with data augmentation techniques without noticeable benefits.
    Model Simplification: Simplified the model structure, yet failed to see an improvement.
    Complexity Increase: Increased the model's complexity, which didn't yield better results.
    Convolutional Layers Adjustment: A model with 4 convolutional layers showed a slight improvement in performance.
    Rescaling Implementation: Added rescaling to the new model, which slightly improved accuracy, presumably due to clearer image representations at a higher scale.
    Dataset Adjustment: Removed the "disgusted" emotion from the dataset due to its underrepresentation, slightly improving the accuracy.

Concluding Attempts

After numerous experiments with different model architectures and strategies, I decided to settle with an accuracy range of 57-59%. According to my research, achieving around 75% accuracy is considered successful for this dataset, highlighting the complexity of emotion detection tasks.
Transition to EfficientNet

Given the limitations faced with initial models and TensorFlow's challenges, I transitioned to PyTorch and decided to explore pretrained models for potentially better outcomes.

    Pretrained Model Exploration: After testing various pretrained models such as VGG16 and different versions of ResNet, none offered the desired improvement until EfficientNet was considered.
    EfficientNet Success: The EfficientNet model stood out, delivering a promising 68.5% accuracy, making it the chosen model for this project. This success underscores the efficiency and accuracy benefits of utilizing advanced, pretrained models in complex classification tasks like emotion detection.

Conclusion

This project's journey from its inception to the successful implementation of EfficientNet exemplifies the iterative and explorative essence of machine learning endeavors. Each phase of development, despite its challenges, has enriched my understanding of model optimization, dataset manipulation, and the nuanced application of deep learning principles. Significantly, this experience has enhanced my proficiency in leveraging GPU capabilities over CPU, marking a substantial improvement in training efficiency. Previously, my projects, constrained by smaller datasets, did not fully exploit the advantages of GPU acceleration. The transition to a more demanding image-based dataset underlined the importance of computational optimization. Additionally, the decision to shift from TensorFlow to PyTorch was driven by TensorFlow's diminishing support for native Windows environments. This transition not only facilitated my model development process but also provided a valuable learning opportunity in adapting to and navigating the evolving landscape of deep learning frameworks.

Harlan Ferguson
101133838