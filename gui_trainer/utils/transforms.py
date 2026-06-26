from PIL import Image
import torchvision.transforms as T


class EEGSpectrogramTransform:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.pipeline = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
        return self.pipeline(img)


class AudioSpectrogramTransform:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.pipeline = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
        return self.pipeline(img)


def get_default_eeg_transform(image_size: int = 224):
    return EEGSpectrogramTransform(image_size)


def get_default_audio_transform(image_size: int = 224):
    return AudioSpectrogramTransform(image_size)
