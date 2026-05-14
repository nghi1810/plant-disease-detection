plant-disease-detection
AI-based plant disease detection using deep learning and computer vision
<img width="1257" height="738" alt="78cfaf46245aa504fc4b" src="https://github.com/user-attachments/assets/fcd83a12-8baa-4658-8b9f-dcc36dd23b84" />
<img width="2560" height="1250" alt="7062b5b13eadbff3e6bc" src="https://github.com/user-attachments/assets/368aa1a9-bcec-43e3-ae63-8fa3aae9ecc7" />

Data Collection
The dataset was collected from multiple plant species including grape, lime, apricot blossom (hoa mai), and rose leaves, covering both healthy and diseased conditions. This ensured a diverse and balanced dataset suitable for training a robust deep learning model in plant disease classification.
<div style="display:flex; gap:10px; flex-wrap:wrap;">
  <img width="200" src="https://github.com/user-attachments/assets/5bc381f2-dfac-4aae-b792-176e32804c07" />
  <img width="200" src="https://github.com/user-attachments/assets/d9e4e381-2dd8-4e22-a7e1-787c62f37574" />
  <img width="200" src="https://github.com/user-attachments/assets/54de03cc-9aa3-447a-8633-ff204a0e3b6b" />
  <img width="200" src="https://github.com/user-attachments/assets/ee841206-732d-4a2f-a185-4dcd91272610" />
</div>


Data Filtering and Organization
Raw images were carefully filtered to remove low-quality, blurry, duplicate, and irrelevant samples. After cleaning, the dataset was organized into structured folders based on plant type and disease categories to maintain consistency and improve data usability.


Data Preprocessing
Preprocessing techniques were applied to standardize the dataset, including image resizing to a fixed resolution, normalization of pixel values, and background noise reduction. These steps helped improve model stability and training efficiency.
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/f9f77b7d-2672-4c7e-9bbe-aeb0a07a0d45" />
<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/25e7e658-11d7-4804-aa98-9617cb76a9fb" />


Dataset Caching
A caching mechanism was implemented to store preprocessed images in a serialized format. This reduced redundant preprocessing during training and significantly improved data loading speed, especially for large datasets.

Data Augmentation
To enhance model generalization, various augmentation techniques were applied such as rotation, horizontal and vertical flipping, zooming, shifting, and brightness/contrast adjustment. These transformations helped the model become more robust to real-world variations.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/f378385f-a6a4-45d8-82b6-2fbb721053a5" />


Model Development
A convolutional neural network (CNN)-based architecture was developed for plant disease detection. The model was designed to extract hierarchical features from leaf images and perform multi-class classification across different plant diseases.
<img width="753" height="352" alt="image" src="https://github.com/user-attachments/assets/5cfac86b-dd79-45ab-a590-7e3cdc46fbd0" />


Training and Evaluation
The model was trained using Python with TensorFlow/PyTorch frameworks. Performance was evaluated using standard metrics such as accuracy and loss, and the model was iteratively optimized through hyperparameter tuning to achieve better classification results.
