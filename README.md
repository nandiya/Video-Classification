# Video-Classification
In this project, I try to make program that could classify object in video using CNN and try to classify movement using CNN+LSTM

List of movements that this program could recognize:
- TAKE     == Taking an object from shelf
- CHECK    == Checking an object or nothing from shelf
- TAKEBACK == Taking and then Returning an object (the same object) to shelf

List of objects that this program could recognize:
- Candy       - Chips   - Nothing   - Sponge
- Cocacola    - Javana  - Oreo      - Sprite
- Chocopie    - Nextar  - Pokka

**  Note : All video are taken using Handphone camera with 720x1280 resolution
    Sum of videos each class: 
    Take        = 205
    CHECK       = 221
    TAKEBACK    = 225     +
    -----------------------
    Total         651


    Chips       = 48
    Cocacola    = 96
    Chocopie    = 41
    Candy       = 16
    Javana      = 97
    Nothing     = 75
    Nextar      = 21
    Pokka       = 82
    Oreo        = 62
    Sponge      = 43       
    Sprite      = 70       +
    --------------------------
    Total         651

How to do the same classification:
1. Run Vid2frame.py. This code will turn generate every video in dataset you have into frames
2. Split every frames folder into training datasets and testing datasets. Since i have unbalanced dataset for my object recognition i      pick random video for each class in obejct recognition (If i use train test split, there's a chance where not all classes will be in    training or testing datasets.
3. Run Temporal Preporcessing.py and Spatial preprocessing.py. the code for spatial preprocessing is quite similar with temporal            preprocessing. You just have to change some codes in temporal preprocessing.py to do spatial preprocessing. Temporal preprocessing      is used for making matrix of optical flow for each video. While spatial preprocessing is used for making matrix of frames by            stacking 10 frames into one stack.
   ** Note: with limited GPU resource that I have, I have to process dataset per each class. 
4. Run Training Movement.py and Training Object.py. 
5. Model and weights that you get from running training movement. py and training object.py can be used to predict the video datasets in    testing data. Before predicting testing data, you have to do the same preprocessing in step to for testing data. You just have to       change the location in source code of temporal preprocessing.py and spatial preprocessing. py into your testing data location.

Result:
Movement Recognition : 76.92 % by succesfully predicting 150 of 195 videos.
Object Recognition   : 55.90 % by succesfully predicting 109 of 195 videos.
