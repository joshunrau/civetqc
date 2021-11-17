def load_saved_model(filepath: str):
    with open(filepath, 'rb') as f:
        return pickle.load(f)