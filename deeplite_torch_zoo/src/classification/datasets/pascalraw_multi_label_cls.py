class PASCALRAWMultiLabelCls(Dataset):
    def __init__(self,
                 root,
                 split: str = "train",
                 transform = None,
                 split_percentage=0.8,
                 ):
        '''
        root: Assuming the tiff files are grouped into folders of corresponding class,
              the root director should point to the parent folder of these folders.
        '''
        self.data_dir = root
        self.transform = transform
        self.split = split
        self.image_files = glob.glob(os.path.join(root, "**/*.tiff"), recursive=True)
        np.random.shuffle(self.image_files)
        num_train_images = int(len(self.image_files)* split_percentage)

        if split == "train":
            self.image_files = self.image_files[:num_train_images]
        else:
            self.image_files = self.image_files[num_train_images:]

        self.classes = set()

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = rawpy.imread(image_path)
        raw_image = image.raw_image
        raw_image = raw_image.astype(np.float16)

        if self.transform:
            raw_image = self.transform(raw_image)
        else:
            image = ToTensor()(raw_image)

        
        label = Path(image_path).parent.stem
        return image, label

    def __len__(self) -> int:
        return len(self.image_files)