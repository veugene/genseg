export DR_256="/data/lisa/data/beckhamc/dr-data/train-trim-256/"
export DR_FOLDERS="/data/milatmp1/beckhamc/tmp_data/dr-data/"

# With pytorch-env-py36 I get the following:
# RuntimeError: To use MKL 2018 with Theano you MUST set "MKL_THREADING_LAYER=GNU" in your environement
# We're not using Theano but we are using Keras datagen which relies on Theano (if that is selected for
# the backend). So I need to actually rip out that data augmentation code so that it's standalone from
# Keras.
export MKL_THREADING_LAYER=GNU
