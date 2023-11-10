from .model import FastSAM
import numpy as np
from PIL import Image
import clip
from typing import Optional, List, Tuple, Union


class FastSAMDecoder:
    def __init__(
        self,
        model: FastSAM,
        device: str='cpu',
        conf: float=0.4, 
        iou: float=0.9,
        imgsz: int=1024,
        retina_masks: bool=True,
        ):
        self.model = model
        self.device = device
        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.image = None
        self.image_embedding = None
        
    def run_encoder(self, image):
        if isinstance(image,str):
            image =  np.array(Image.open(image))
        self.image = image
        image_embedding = self.model(
            self.image, 
            device=self.device, 
            retina_masks=self.retina_masks, 
            imgsz=self.imgsz, 
            conf=self.conf, 
            iou=self.iou
            )
        return image_embedding[0].numpy()

    def run_decoder(
            self,
            image_embedding, 
            point_prompt: Optional[np.ndarray]=None,
            point_label: Optional[np.ndarray]=None,
            box_prompt: Optional[np.ndarray]=None,
            text_prompt: Optional[str]=None,
            ) -> np.ndarray:
        self.image_embedding = image_embedding
        if point_prompt is not None:
            return self.point_prompt(points=point_prompt, pointlabel=point_label)
        elif box_prompt is not None:
            return self.box_prompt(bbox=box_prompt)
        elif text_prompt is not None:
            return self.text_prompt(text=text_prompt)
        else:
            return None

    def box_prompt(self, bbox):
        assert (bbox[2] != 0 and bbox[3] != 0)
        masks = self.image_embedding.masks.data
        target_height = self.image.shape[0]
        target_width = self.image.shape[1]
        h = masks.shape[1]
        w = masks.shape[2]
        if h != target_height or w != target_width:
            bbox = [
                int(bbox[0] * w / target_width),
                int(bbox[1] * h / target_height),
                int(bbox[2] * w / target_width),
                int(bbox[3] * h / target_height), ]
        bbox[0] = max(round(bbox[0]), 0)
        bbox[1] = max(round(bbox[1]), 0)
        bbox[2] = min(round(bbox[2]), w)
        bbox[3] = min(round(bbox[3]), h)

        # IoUs = torch.zeros(len(masks), dtype=torch.float32)
        bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

        masks_area = np.sum(masks[:, bbox[1]:bbox[3], bbox[0]:bbox[2]], axis=(1, 2))
        orig_masks_area = np.sum(masks, axis=(1, 2))

        union = bbox_area + orig_masks_area - masks_area
        IoUs = masks_area / union
        max_iou_index = np.argmax(IoUs)

        return np.array([masks[max_iou_index].cpu().numpy()])

    def point_prompt(self, points, pointlabel):  # numpy 

        masks = self._format_results(self.image_embedding[0], 0)
        target_height = self.image.shape[0]
        target_width = self.image.shape[1]
        h = masks[0]['segmentation'].shape[0]
        w = masks[0]['segmentation'].shape[1]
        if h != target_height or w != target_width:
            points = [[int(point[0] * w / target_width), int(point[1] * h / target_height)] for point in points]
        onemask = np.zeros((h, w))
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        for i, annotation in enumerate(masks):
            mask = annotation['segmentation'] if type(annotation) == dict else annotation
            for i, point in enumerate(points):
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                    onemask[mask] = 1
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                    onemask[mask] = 0
        onemask = onemask >= 1
        return np.array([onemask])
    
    def _format_results(self, result, filter=0):
        annotations = []
        n = len(result.masks.data)
        for i in range(n):
            mask = result.masks.data[i] == 1.0

            if np.sum(mask) < filter:
                continue
            annotation = {
                'id': i,
                'segmentation': mask,
                'bbox': result.boxes.data[i],
                'score': result.boxes.conf[i],
            }
            annotation['area'] = annotation['segmentation'].sum()
            annotations.append(annotation)
        return annotations
