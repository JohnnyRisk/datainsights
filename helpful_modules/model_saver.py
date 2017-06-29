# serialize weights to HDF5
# serialize model to JSON
def save_model_w_weights(model, name):
    #serialize model to JSON
    model_json = model.to_json()
    with open('./autoencoder_models/'+name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('./autoencoder_models/'+name+".h5")
    print("Saved model to disk")

# load json and create model
def load_model_w_weights(model_name):
    from keras.models import model_from_json
    json_file = open('./autoencoder_models/'+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('./autoencoder_models/'+model_name+".h5")
    print("Loaded model from disk")
    return loaded_model
