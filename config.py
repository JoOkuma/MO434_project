import os

TACO_DIR = '/tmp/TACO'

class Config:
    @staticmethod
    def images_dir() -> str:
        return os.path.join(TACO_DIR, 'data')

    @staticmethod
    def annot_path() -> str:
        return os.path.join(TACO_DIR, 'data/annotations.json')

    @staticmethod
    def class10_config() -> str:
        return os.path.join(TACO_DIR, 'detector/taco_config/map_10.csv')

    @staticmethod
    def class4_config() -> str:
        return os.path.join(TACO_DIR, 'detector/taco_config/map_4.csv')

