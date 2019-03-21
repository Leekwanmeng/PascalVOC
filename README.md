# PascalVOC

## Setup

1. Clone this repository

2. Install and extract Pascal VOC2012 data
```chmod +x setup.sh```
```./setup.sh```

3. Install package requirements
```pip install requirements.txt```

##Training the model

1. To run the train model script, `python3 ./models/train_model.py`

##Testing the model

1. Set the `model_path` variable in `test_model.py` to the path of the saved model

2. To run the test model script, `python3 ./models/test_model.py`

##Running the GUI

1. To start the GUI interface, `python3 ./app/main.py`

2. To predict a sample image, you can use the `sample_image.jpg` provided.
