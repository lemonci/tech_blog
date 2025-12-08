---
description: Translation based on https://github.com/KwanWaiPang/Photo-SLAM_comment/
---

# Photo-SLAM code brief

Photo-SLAM\
├── cfg                            # This folder includes the config files for different settings\
├── cuda\_rasterizer     # This folder includes the rasterization module from the original 3DGS \
├── examples                # This folder is the demo entry point of SLAM system\
&#x20;                                             It is used for reading the data, creating pointers and mapping\
├── include                     # Head files\
├── ORB-SLAM3\
├── scripts                      # Scripts to test running and evaluation\
├── src                             # 3D Gaussian related scripts\
├── third\_party              # Relied packages: colmap, simple-knn(from 3D Gaussian) and tinyply\
└── viewer                       # Thread of visualizer<br>

WLOG, let us start with [example/tum\_rgbd.cpp](https://github.com/HuajianUP/Photo-SLAM/blob/main/examples/tum_rgbd.cpp):

It includes functions `main`, `LoadImages`(to read images), `saveTrackingTime` (to save trajectory) and `saveGpuPeakMemoryUsage` (to save the peak of VRAM usage).

What does function `main` do? It checks the input parameters, input directories and load the images. The most important is to establish the SLAM thread `pSLAM` , the 3D gaussian mapping thread `pGausMapper`  and Gaussian viewer thread `pViewer` . `pSLAM` works as the input parameter of `pGausMapper` , connecting ORB-SLAM and 3D Gaussian mapping.

Specifically, the code did the following stuff:

1. Check the parameter and set the output directory.
2. Check the input image directory and call `LoadImages` to read the images.

```
LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
```

3. Assure the amounts of depth images and RDB images are the same. Check to use CPU or GPU.

After everything is ready, we can start to create the SLAM system. The following operations are: 1. to create SLAM system. 2. to create 3D Gaussian mapping system and 3. to create 3D Gaussian viewer.

```
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // Create the pointer pSLAM to point to the SLAM system
    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::RGBD);
    float imageScale = pSLAM->GetImageScale();

    // Create GaussianMapper
    // Create the pointer pGuasMapper to point to the 3D Gaussian Mapper,
    // the input parameter is the SLAM system, configs for 3D Gaussian 
    // and output directory
    std::filesystem::path gaussian_cfg_path(argv[3]);
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(
            pSLAM, gaussian_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    // If the GUI config is true, we start the viewer thread and
    // pass SLAM system pointer and 3D Gaussian Mapper pointer as input
    if (use_viewer)
    {  
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }
```
