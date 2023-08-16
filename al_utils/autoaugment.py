from PIL import Image, ImageEnhance, ImageOps
import PIL.ImageDraw as ImageDraw
import numpy as np
import random


class RandAugmentPolicy(object):
    """Randomly choose one of the best 25 Sub-policies on CIFAR10.
    Example:
    >>> policy = RandAugmentPolicy()
    >>> transformed = policy(image)
    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     RandAugmentPolicy(),
    >>>     transforms.ToTensor()])
    """

    # I change fill color from (128, 128, 128) to (0, 0, 0)
    def __init__(self, fillcolor=(0, 0, 0), N=1, M=5):
        self.policies = [
            "invert",
            "autocontrast",
            "equalize",
            "rotate",
            "solarize",
            "color",
            "posterize",
            "contrast",
            "brightness",
            "sharpness",
            "shearX",
            "shearY",
            "translateX",
            "translateY",
            "cutout",
        ]
        self.N = N
        self.M = M

    def __call__(self, img):
        """Applies RandAugment on input image.

        Args:
            img: Input image

        Returns:
            np.ndarray: Augmented image based on N and M hyper-parameters chosen for RA.
        """

        choosen_policies = np.random.choice(self.policies, self.N)
        for policy in choosen_policies:
            subpolicy_obj = SubPolicy(operation=policy, magnitude=self.M)
            img = subpolicy_obj(img)

        return img
        # return self.policies[policy_idx](img)

    def __repr__(self):
        return f"RandAugment Policy with Cutout where N: {self.N} and M: {self.M}"


class SubPolicy(object):
    def __init__(self, operation, magnitude, fillcolor=(128, 128, 128), MAX_PARAM=10):
        ranges = {
            "shearX": np.linspace(0, 0.3, MAX_PARAM),
            "shearY": np.linspace(0, 0.3, MAX_PARAM),
            "translateX": np.linspace(0, 150 / 331, MAX_PARAM),
            "translateY": np.linspace(0, 150 / 331, MAX_PARAM),
            "rotate": np.linspace(0, 30, MAX_PARAM),
            "color": np.linspace(0.0, 0.9, MAX_PARAM),
            "posterize": np.round(np.linspace(8, 4, MAX_PARAM), 0).astype(np.int64),
            "solarize": np.linspace(256, 0, MAX_PARAM),
            "contrast": np.linspace(0.0, 0.9, MAX_PARAM),
            "sharpness": np.linspace(0.0, 0.9, MAX_PARAM),
            "brightness": np.linspace(0.0, 0.9, MAX_PARAM),
            "autocontrast": [0] * MAX_PARAM,
            "equalize": [0] * MAX_PARAM,
            "invert": [0] * MAX_PARAM,
            "cutout": np.linspace(0.0, 0.8, MAX_PARAM),
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (0,) * 4), rot
            ).convert(img.mode)

        def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
            assert 0.0 <= v <= 0.8
            if v <= 0.0:
                return img

            v = v * img.size[0]

            return CutoutAbs(img, v)

        def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
            # assert 0 <= v <= 20
            if v < 0:
                return img
            w, h = img.size
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)

            x0 = int(max(0, x0 - v / 2.0))
            y0 = int(max(0, y0 - v / 2.0))
            x1 = min(w, x0 + v)
            y1 = min(h, y0 + v)

            xy = (x0, y0, x1, y1)
            # color = (125, 123, 114)
            color = (0, 0, 0)
            img = img.copy()
            ImageDraw.Draw(img).rectangle(xy, color)
            return img

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
            "cutout": lambda img, magnitude: Cutout(img, magnitude),
        }

        self.operation = func[operation]
        self.magnitude = ranges[operation][magnitude]

    def __call__(self, img):
        # if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        img = self.operation(img, self.magnitude)
        return img


class SplitAugmentPolicy(object):
    def __init__(self, N=1, M=5, slices=4):
        self.policies = [
            "invert",
            "autocontrast",
            "equalize",
            "rotate",
            "solarize",
            "color",
            "posterize",
            "contrast",
            "brightness",
            "sharpness",
            "shearX",
            "shearY",
            "translateX",
            "translateY",
            "cutout",
        ]
        self.N = N
        self.M = M
        self.slices = slices
        # print(f"Split Augment Object initialized: N: {self.N}, M: {self.M} and slices: {self.slices}")

    def __call__(self, img):
        # policy_idx = random.randint(0, len(self.policies) - 1)
        w, h = img.size
        img = np.array(img)

        if self.slices == 4:
            img_1 = Image.fromarray(img[0:16, 0:16, :])
            img_2 = Image.fromarray(img[0:16, 16:, :])
            img_3 = Image.fromarray(img[16:, 0:16, :])
            img_4 = Image.fromarray(img[16:, 16:, :])

            imgess = [img_1, img_2, img_3, img_4]
        elif self.slices == 2:
            img_1 = Image.fromarray(img[:, 0:16, :])
            img_2 = Image.fromarray(img[:, 16:, :])

            imgess = [img_1, img_2]
        else:
            raise NotImplementedError

        for i, img in enumerate(imgess):
            choosen_policies = np.random.choice(self.policies, self.N)
            for policy in choosen_policies:
                subpolicy_obj = SubPolicy(operation=policy, magnitude=self.M)
                img = subpolicy_obj(img)
                imgess[i] = np.array(img)

        temp_img = np.zeros(shape=(w, h, 3))
        if self.slices == 4:
            temp_img[0:16, 0:16, :] = imgess[0]
            temp_img[0:16, 16:, :] = imgess[1]
            temp_img[16:, 0:16, :] = imgess[2]
            temp_img[16:, 16:, :] = imgess[3]
        elif self.slices == 2:
            temp_img[:, 0:16, :] = imgess[0]
            temp_img[:, 16:, :] = imgess[1]
        else:
            raise NotImplementedError

        temp_img = Image.fromarray(temp_img.astype(np.uint8))
        return temp_img

    def __repr__(self):
        return "SplitAugment CIFAR Policy with Cutout"
