import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from models.model_single import ModelEmb
import pickle


def sam_call(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings_none,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    return low_res_masks


def segment_image(image, model):
    original_size = image.shape[:2]

    Idim = 512
    image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    image_tensor = transform.apply_image_torch(image_tensor)

    input_images = transform.preprocess(image_tensor).unsqueeze(dim=0)
    input_images = image_tensor.unsqueeze(dim=0).cuda()

    orig_imgs_small = F.interpolate(
        input_images, (Idim, Idim), mode="bilinear", align_corners=True
    )
    dense_embeddings = model(orig_imgs_small)

    mask = sam_call(input_images, sam, dense_embeddings)
    input_size = image_tensor.shape[1:]
    mask = sam.postprocess_masks(
        mask, input_size=input_size, original_size=original_size
    )
    mask = mask.squeeze().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (255 * mask).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def click_event(event, x, y, flags, param):
    global image, point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        mask = segment_image(image, point)
        cv2.imwrite("tmp.jpg", mask)
        cv2.imshow("Mask", mask)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-i", "--input_image", help="Path to input image", required=True
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Path to prompt encoder model checkpoint",
        required=True,
    )
    parser.add_argument(
        "-o", "--output_image", help="Path to output image", required=True
    )
    args = vars(parser.parse_args())
    # Load your image
    image = cv2.cvtColor(
        cv2.imread(args["input_image"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )  # cv2.imread(args["input_image"])
    point = None
    sam_args = {
        "sam_checkpoint": "cp/sam_vit_b.pth",
        "model_type": "vit_b",
        "generator_args": {
            "points_per_side": 8,
            "pred_iou_thresh": 0.95,
            "stability_score_thresh": 0.7,
            "crop_n_layers": 0,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 0,
            "point_grids": None,
            "box_nms_thresh": 0.7,
        },
        "gpu_id": 0,
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    sam = sam_model_registry[sam_args["model_type"]](
        checkpoint=sam_args["sam_checkpoint"]
    ).cuda()
    transform = ResizeLongestSide(1024)

    # cv2.imshow("Image", image)
    emb_def_args = {
        "depth_wise": False,
        "order": 68,
    }
    model = torch.load(args["checkpoint"])  # ModelEmb(args=emb_def_args).to(device)

    mask = segment_image(image, model)

    print("mask", mask.shape, "image", image.shape)

    # overlay_color = (0, 0, 255)
    # overlay = np.zeros_like(image)
    cv2.imwrite("mask_" + args["output_image"], mask)

    overlay = (np.array(mask) * np.array([0, 0, 1])).astype(image.dtype)
    print("overlay", overlay.shape)
    # overlay[mask[:, :, 0] > 50] = overlay_color
    alpha = 0.4  # Transparency factor
    output_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    cv2.imwrite(args["output_image"], output_image)
