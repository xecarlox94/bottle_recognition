import mlflow
from ultralytics import YOLO
import subprocess


#from roboflow import Roboflow
#rf = Roboflow(api_key="llgtjSRbLz3hdjjaUEfb")
#project = rf.workspace("erkan-unal").project("detect-o9dby")
#
#dataset = project.version(1).download("yolov8")
#
#subprocess.call(['/bin/bash', 'sed -i "s/detect-1/../g" detect-1/data.yaml'])


model = YOLO('yolov8n.pt')

dataset_yaml = './detect-1/data.yaml'


with mlflow.start_run() as run:
    model.train(data=dataset_yaml, epochs=1, imgsz=640, device=0)


    #########################################################
    #
    #
    # TEMPORARY: local imports and schema signature inference
    #
    #
    #########################################################
    # temporary local imports
    # TODO: eventually move them to start of function
    from os.path import join
    from mlflow.models import infer_signature
    from ultralytics.cfg import RUNS_DIR
    from ultralytics.nn.tasks import (
        DetectionModel,
        PoseModel,
        SegmentationModel,
        ClassificationModel
    )

    trainer = model.trainer
    #########################################################
    #
    #
    # DATASET Logic     (X, y)
    #
    #
    #########################################################
    # TODO: switch statement for each task

    from mlflow.models import ModelSignature
    import numpy as np
    from mlflow.models import ModelSignature, infer_signature
    from mlflow.types.schema import Schema, TensorSpec
    input_schema = Schema(
        [
            TensorSpec(np.dtype(np.float64), (-1, 640, 640, 3)),
        ]
    )

    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 2))])

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)


    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    dataset = trainer.train_loader.dataset
    dataloader = trainer.get_dataloader(dataset.data['train'])


    # https://docs.ultralytics.com/modes/predict/#working-with-results
    result = next(iter(dataloader))


    # https://docs.ultralytics.com/modes/train/#key-features-of-train-mode
    yolo_model = YOLO(trainer.best)

    X_train = result['img']

    img = result['im_file'][0]
    rst = yolo_model(img)

    from torchvision import transforms

    import numpy as np

    convert_tensor = transforms.ToTensor()

    # original image
    img_npy = rst[0].orig_img

    #####
    img_this = convert_tensor(img_npy).cuda()
    tensor_image = (img_this * 255).float()
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)


    # possible y
    y_train = trainer.model(tensor_image)


    # model_type = type(trainer.model)
    # if model_type is DetectionModel:
    #     y_train = result['bboxes']
    #     # I think we need the classifications values, along bboxes, as a tuple, in y (in detection)

    # elif model_type is PoseModel:
    #     y_train = result['keypoints']

    # elif model_type is ClassificationModel:
    #     y_train = result['probs']

    # elif model_type is SegmentationModel:
    #     y_train = result['masks']


    y_train = y_train.detach().cpu().numpy()


    # Important
    # https://mlflow.org/docs/latest/models.html#tensor-based-signature-example

    sig = infer_signature(
        # X_train.detach().cpu().numpy(),
        tensor_image.cpu().numpy(),
        np.asarray((y_train[0].data).detach().cpu()),
        params={
            "param_test": 123
        }
    )


    mlflow.pytorch.log_model(
        trainer.model,
        join(RUNS_DIR.name, "Model"),
        signature=signature
    )

    print("end")

    #########################################################
    #
    #
    # Multiple format exporting
    #
    #
    #########################################################
    # TODO: get export format; other formats in a switch statement below. more info: https://docs.ultralytics.com/modes/export/#key-features-of-export-mode
    # TODO: make parameters visible and accessible in runs

