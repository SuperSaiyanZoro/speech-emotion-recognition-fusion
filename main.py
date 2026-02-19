from models.fusion_model import FusionModel
from training.trainer import Trainer

def main():
    print("Speech Emotion Recognition - Fusion Model")
    model = FusionModel()
    trainer = Trainer(model)
    print("Model initialized successfully")

if __name__ == "__main__":
    main()
