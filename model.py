
from transformers import SegformerForSemanticSegmentation



def Model(num_classes):
    model= SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                                                           num_labels=num_classes,
                                                           ignore_mismatched_sizes=True)
    
    
    return model


