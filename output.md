(uav_landing_project) PS H:\landing-system\uav_landing_project> python scripts/train_semantic_model.py --batch-size 8 --epochs 50 --output-dir outputs/test_single_stage
================================================================================
Single-Stage Semantic Segmentation Training
================================================================================
This script trains a model exclusively on the Semantic Drone Dataset.
It uses a pre-trained model and fine-tunes it in a single, robust stage.
--------------------------------------------------------------------------------
wandb: Currently logged in as: debkumar269 (debkumar269-macquarie-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.21.0
wandb: Run data is saved locally in H:\landing-system\uav_landing_project\wandb\run-20250722_234052-oowsp1az
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run train-mmseg_bisenetv2-py2rqnxc
wandb:  View project at https://wandb.ai/debkumar269-macquarie-university/uav-landing-single-stage
wandb:  View run at https://wandb.ai/debkumar269-macquarie-university/uav-landing-single-stage/runs/oowsp1az
Configuration:
  - dataset_path: H:\landing-system\datasets\Aerial_Semantic_Segmentation_Drone_Dataset\dataset\semantic_drone_dataset
  - pretrained_path: H:\landing-system\model_pths\bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth
  - output_dir: outputs/test_single_stage
  - model_type: mmseg_bisenetv2
  - input_size: [512, 512]
  - epochs: 50
  - batch_size: 8
  - learning_rate: 0.0001
  - num_workers: 4
  - wandb_project: uav-landing-single-stage
--------------------------------------------------------------------------------
GPU: NVIDIA GeForce RTX 4060 Ti
   - Memory: 8.59 GB
--------------------------------------------------------------------------------
Loading datasets...
H:\landing-system\uav_landing_project\.venv\Lib\site-packages\albumentations\core\validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
H:\landing-system\uav_landing_project\datasets\semantic_drone_dataset.py:358: UserWarning: Argument(s) 'value' are not valid for transform ShiftScaleRotate    
  A.ShiftScaleRotate(
H:\landing-system\uav_landing_project\datasets\semantic_drone_dataset.py:373: UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise
  A.GaussNoise(var_limit=(10, 50)),
H:\landing-system\uav_landing_project\datasets\semantic_drone_dataset.py:377: UserWarning: Argument(s) 'shift_limit' are not valid for transform OpticalDistortion
  A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
SemanticDroneDataset initialized:
   Split: train (280 samples)
   Classes: 4 landing classes
   Resolution: (512, 512)
   Mapping: enhanced_4_class
SemanticDroneDataset initialized:
   Split: val (60 samples)
   Classes: 4 landing classes
   Resolution: (512, 512)
   Mapping: enhanced_4_class
  - Training samples: 280
  - Validation samples: 60
Creating model: mmseg_bisenetv2
Loading MMSeg BiSeNetV2 weights from: H:\landing-system\model_pths\bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth
 Pretrained weight loading completed:
   Loaded layers: 75
   Adapted layers: 10
   Skipped layers: 221
🔧 Adapted classifier layers:
   decode_head.conv_seg.weight (torch.Size([19, 1024, 1, 1])→torch.Size([4, 1024, 1, 1]))
   decode_head.conv_seg.bias (torch.Size([19])→torch.Size([4]))
   auxiliary_head.0.conv_seg.weight (torch.Size([19, 16, 1, 1])→torch.Size([4, 16, 1, 1]))
   auxiliary_head.0.conv_seg.bias (torch.Size([19])→torch.Size([4]))
   auxiliary_head.1.conv_seg.weight (torch.Size([19, 64, 1, 1])→torch.Size([4, 64, 1, 1]))
   auxiliary_head.1.conv_seg.bias (torch.Size([19])→torch.Size([4]))
   auxiliary_head.2.conv_seg.weight (torch.Size([19, 256, 1, 1])→torch.Size([4, 256, 1, 1]))
   auxiliary_head.2.conv_seg.bias (torch.Size([19])→torch.Size([4]))
   auxiliary_head.3.conv_seg.weight (torch.Size([19, 1024, 1, 1])→torch.Size([4, 1024, 1, 1]))
   auxiliary_head.3.conv_seg.bias (torch.Size([19])→torch.Size([4]))
🏗️ Created MMSeg BiSeNetV2:
   Parameters: 13,019,725 (13,019,725 trainable)
   Model size: ~49.7 MB
   Uncertainty: True

Starting training...


python scripts/train_semantic_model.py --batch-size 16 --num-workers 16 --epochs 50