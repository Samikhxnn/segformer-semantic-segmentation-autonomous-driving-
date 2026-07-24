import torch


def calculate_iou(pred,mask,num_classes):

    pred=torch.argmax(
        pred,
        dim=1
    )


    ious=[]


    for cls in range(num_classes):

        pred_cls=(pred==cls)

        mask_cls=(mask==cls)


        intersection=(
            pred_cls & mask_cls
        ).sum()


        union=(
            pred_cls | mask_cls
        ).sum()


        if union==0:
            continue


        ious.append(
            intersection.float()/union.float()
        )


    return torch.mean(
        torch.stack(ious)
    )