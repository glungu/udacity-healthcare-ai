from keras.models import model_from_json
from keras.utils import plot_model

# read architecture file
model_json = open('xray_final_model.json', 'r')
model_json_content = model_json.read()
model_json.close()

# load model architecture
model = model_from_json(model_json_content)

# load model weights
model.load_weights("xray_classification_model.best.hdf5")

# visualize
plot_model(
    model,
    to_file="model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
)

