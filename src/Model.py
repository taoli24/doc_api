from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, \
    LayoutLMv3ForTokenClassification
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image


class LMModel:
    def __init__(self, checkpoint: Path, labels_count: int):
        self.checkpoint = checkpoint
        self.labels_count = labels_count
        self.feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True, ocr_lang="eng", size=[224, 224])
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
        self.processor = LayoutLMv3Processor(self.feature_extractor, self.tokenizer)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",
                                                                      num_labels=self.labels_count)
        # load check model checkpoint
        check_point = torch.load(self.checkpoint, map_location=torch.device('cpu'))
        self.model.load_state_dict(check_point['model_state_dict'])

    @staticmethod
    def process_image(image_path: Path):
        image = cv2.imread(image_path)
        image_cvtColor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_cvtColor = cv2.cvtColor(image_cvtColor, cv2.COLOR_RGB2GRAY)
        final_image_input = np.array([image_cvtColor, image_cvtColor, image_cvtColor]).transpose(1, 2, 0)
        final_image = Image.fromarray(final_image_input)
        return final_image

    def get_encoding(self, image: Image):
        return self.processor(image,
                              max_length=512,
                              padding="max_length",
                              truncation=True,
                              return_tensors="pt")

    def get_output(self, image: Image):
        self.model.eval()
        encoded_data = self.get_encoding(image)
        outputs = self.model(**encoded_data)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoded_data.bbox.squeeze().tolist()
        return {"bbox": token_boxes, "predictions": predictions}

    def __call__(self, *args, **kwargs):
        return self.get_output(kwargs["image"])
