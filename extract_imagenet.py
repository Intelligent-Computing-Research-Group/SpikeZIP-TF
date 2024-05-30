import os
import tarfile

TRAIN_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar'
TRAIN_DEST_DIR = '/root/autodl-tmp/imagenet/train'
VAL_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar'
VAL_DEST_DIR = '/root/autodl-tmp/imagenet/val'


def extract_train():
    with open(TRAIN_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        for i, item in enumerate(tar):
            cls_name = item.name.strip(".tar")
            a = tar.extractfile(item)
            b = tarfile.open(fileobj=a, mode="r:")
            e_path = "{}/{}/".format(TRAIN_DEST_DIR, cls_name)
            if not os.path.isdir(e_path):
                os.makedirs(e_path)
            print("#", i, "extract train dateset to >>>", e_path)
            b.extractall(e_path)
            # names = b.getnames()
            # for name in names:
            #     b.extract(name, e_path)


def extract_val():
    with open(VAL_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        if not os.path.isdir(VAL_DEST_DIR):
            os.makedirs(VAL_DEST_DIR)
        print("extract val dateset to >>>", VAL_DEST_DIR)
        names = tar.getnames()
        for name in names:
            tar.extract(name, VAL_DEST_DIR)


if __name__ == '__main__':
    extract_train()
    extract_val()

