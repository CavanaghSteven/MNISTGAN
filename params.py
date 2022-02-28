class Params:
    def __init__(self):
        self.num_epochs: int = 100000
        self.batch_size: int = 8

        self.img_dim: tuple = (28, 28, 1)
        self.model_name: str = 'testering'
        self.latent_dim: int = 128

        self.weight_dir_ext: str = 'weights'
        self.imgs_dir_ext: str = 'imgs'
