# modul ini untuk compare semua proses
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    
    # Preprocessing
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    # Transformation for training model
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    # Model training, evaluating and save object
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

if __name__ == "__main__":
    main()