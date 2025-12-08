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

## Main Loop

WLOG, let us start with [example/tum\_rgbd.cpp](https://github.com/HuajianUP/Photo-SLAM/blob/main/examples/tum_rgbd.cpp):

It includes functions `main`, `LoadImages`(to read images), `saveTrackingTime` (to save trajectory) and `saveGpuPeakMemoryUsage` (to save the peak of VRAM usage).

What does function `main` do? It checks the input parameters, input directories and load the images. The most important is to establish the SLAM thread `pSLAM` , the 3D gaussian mapping thread `pGausMapper`  and Gaussian viewer thread `pViewer` . `pSLAM` works as the input parameter of `pGausMapper` , connecting ORB-SLAM and 3D Gaussian mapping.

Specifically, the code did the following stuff:

1. Check the parameter and set the output directory.
2. Check the input image directory and call `LoadImages` to read the images.

{% code fullWidth="true" %}
```
LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
```
{% endcode %}

3. Assure the amounts of depth images and RDB images are the same. Check to use CPU or GPU.

After everything is ready, we can start to create the SLAM system. The following operations are: \
1\. to create SLAM system. \
2\. to create 3D Gaussian mapping system and \
3\. to create 3D Gaussian viewer.

{% code fullWidth="true" %}
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
{% endcode %}

4. Then we output the system information.\
   Load the RGB and depth information and check if they are valid.\
   Scale RGB and depth image according to `imageScale`.\
   Timer (`chrono` is the new timing method in C++ 11)\
   Import RGB, depth image and frame time into `pSLAM` system.

{% code fullWidth="true" %}
```
// (The exiciting) Main loop.
// Load all images in sequence and deal with them.
    cv::Mat imRGB, imD;
    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown())
            break;
        // Read image and depthmap from file
        imRGB = cv::imread(std::string(argv[4]) + "/" + vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        cv::cvtColor(imRGB, imRGB, CV_BGR2RGB);
        imD = cv::imread(std::string(argv[4]) + "/" + vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        // Handle the empty images
        if (imRGB.empty())
        {
            std::cerr << std::endl << "Failed to load image at: "
                      << std::string(argv[4]) << "/" << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }
        if (imD.empty())
        {
            std::cerr << std::endl << "Failed to load depth image at: "
                      << std::string(argv[4]) << "/" << vstrImageFilenamesD[ni] << std::endl;
            return 1;
        }

        // Scale the images
        if (imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        // Timing
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        pSLAM->TrackRGBD(imRGB, imD, tframe, std::vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni]);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }
```
{% endcode %}

After the main loop ends, close the SLAM system and visualization system, output the GPU peak usage and  tracking information. End the main thread.

&#x20;`TrackRGBD` is the ORB tracking. And we integrate orb tracking\&mapping and 3D Gaussian mapping through:

```
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(
            pSLAM, gaussian_cfg_path, output_dir, 0, device_type);
```

Then what does [`GaussianMapper`](https://github.com/HuajianUP/Photo-SLAM/blob/main/src/gaussian_mapper.cpp) do?

## src/gaussian\_mapper.cpp

This class include many functions but the main content is the constructor, run thread, Gaussian training and keyframe handler.

The constructor doesn't only handle the input parameters but also operate on its private variables. Set up the running device and camera model, initialize 3D Gaussian scene and set up the sensor types.

### The constructor of GaussianMapper

{% code fullWidth="true" %}
```
//
GaussianMapper::GaussianMapper(
    std::shared_ptr<ORB_SLAM3::System> pSLAM,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,//random seed is 0
    torch::DeviceType device_type)
    : pSLAM_(pSLAM),//treat is as if already got the data from orbslam
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    // In standard C++ library, this code is used to set up the random seed.
      std::srand(seed);/
    // This is the random seed set up for PyTorch/litorch.
    torch::manual_seed(seed);

    // Device(set up according to the input device)
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    // Create the output directory for results
    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    // Read the config file
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    // Set up the background color
    std::vector<float> bg_color;
    if (model_params_.white_background_)//If it is white
        bg_color = {1.0f, 1.0f, 1.0f};
    else //Otherwise it is black
        bg_color = {0.0f, 0.0f, 0.0f};
    // Convert bg_color to a tensor and save it in the variable named background_
    // Here we use torch::TensorOptions() to specify the options for the tensor
    // including the data type to torch::kFloat32 and the device type
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // Initialize scene and model including the SH parameter and 
    // Initializ 3DGS tensor
    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    scene_ = std::make_shared<GaussianScene>(model_params_);

    // Mode (If there is no SLAM results then return)
    if (!pSLAM) {
        // NO SLAM
        return;
    }

    // Specify tensor types
    switch (pSLAM->getSensorType())
    {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULAR;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREO;
        this->stereo_baseline_length_ = pSLAM->getSettings()->b();
        this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
            this->stereo_min_disparity_,
            this->stereo_num_disparity_);
        this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
        stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    }
    break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
    {
        this->sensor_type_ = RGBD;
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    // Cameras
    // TODO: not only monocular
    auto settings = pSLAM->getSettings();
    cv::Size SLAM_im_size = settings->newImSize();

    //Get the undisort parameters of the camera
    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );

    //Get all cameras and the intrinsics accordingly
    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    for (auto& SLAM_camera : vpCameras) {
        Camera camera;//Create a camera class

        //Get the id of the camera
        camera.camera_id_ = SLAM_camera->GetId();

        //Get the type of the camera to fetch the instrinc matrix
        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            // Old K, i.e. K in SLAM
            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );

            // camera.width_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.width
            //                                              : graphics_utils::roundToIntegerMultipleOf16(
            //                                                    undistort_params.old_size_.width);
            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;

            // camera.height_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.height
            //                                               : graphics_utils::roundToIntegerMultipleOf16(
            //                                                     undistort_params.old_size_.height);
            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
            camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
            camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
            camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_<float>(3, 3)
                    << camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

            // Undistortion
            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
                undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                       cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

            if (this->sensor_type_ == STEREO) {
                camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
                if (this->stereo_Q_.cols != 4) {
                    this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
                    this->stereo_Q_.setTo(0.0f);
                    this->stereo_Q_.at<float>(0, 0) = 1.0f;
                    this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at<float>(1, 1) = 1.0f;
                    this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
                }
            }
        }
        else if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
            camera.setModelId(Camera::CameraModelType::FISHEYE);
        }
        else {
            camera.setModelId(Camera::CameraModelType::INVALID);
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);//In the end, we place the camera into the scene
    }
}
```
{% endcode %}

### `void run()`&#x20;

&#x20;`void run()` is the main main process, read the camera pose and point cloud, prepare the images of multiple resolutions and call the train function trainForOneIteration()



{% code fullWidth="true" %}
```
void GaussianMapper::run()
{
    // First loop: Initial gaussian mapping
    // Initially, we use a loop to initialize the 3D Gaussian mapping process and single iteration training
    // The loop will end once the initialization is executed and the first training iteration is finished.
    while (!isStopped()) {
        // Check conditions for initial mapping
        // We need ORBSLAM is running and the number of keyframes is over a certain amount
        if (hasMetInitialMappingConditions()) {

            //Clean the map of orbslam3
            pSLAM_->getAtlas()->clearMappingOperation();

            // Get initial sparse map from orbSLAM's current map
            auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
            std::vector<ORB_SLAM3::MapPoint*> vpMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                //Get keyframes
                vpKFs = pMap->GetAllKeyFrames();
                //Get map points
                vpMPs = pMap->GetAllMapPoints();

                //Iterate all map points to get their 3D coordinates and colors. Then place them into scene_
                for (const auto& pMP : vpMPs){
                    Point3D point3D;
                    auto pos = pMP->GetWorldPos();
                    point3D.xyz_(0) = pos.x();
                    point3D.xyz_(1) = pos.y();
                    point3D.xyz_(2) = pos.z();
                    auto color = pMP->GetColorRGB();
                    point3D.color_(0) = color(0);
                    point3D.color_(1) = color(1);
                    point3D.color_(2) = color(2);

                    //Place the points and colors into cached_point_cloud_
                    scene_->cachePoint3D(pMP->mnId, point3D);
                }

                //Then iterate all keyframes
                for (const auto& pKF : vpKFs){
                    //Create a new Gaussian keyframe based on the keyframe id and current iteration number
                    std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(pKF->mnId, getIteration());
                    //The closet and farest depth
                    new_kf->zfar_ = z_far_;
                    new_kf->znear_ = z_near_;

                    //Get pose
                    auto pose = pKF->GetPose();//获取关键帧对应的pose
                    //Set pose
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());

                    // Get image information
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    try {
                        // Camera
                        Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                        new_kf->setCameraParams(camera);

                        // Image (left if STEREO)
                        cv::Mat imgRGB = pKF->imgLeftRGB;
                        if (this->sensor_type_ == STEREO)
                            imgRGB_undistorted = imgRGB;
                        else
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                        // Auxiliary Image
                        cv::Mat imgAux = pKF->imgAuxiliary;
                        if (this->sensor_type_ == RGBD)
                            camera.undistortImage(imgAux, imgAux_undistorted);
                        else
                            imgAux_undistorted = imgAux;

                        new_kf->original_image_ =
                            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                        new_kf->img_filename_ = pKF->mNameFile;
                        new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                        new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                        new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                    }
                    catch (std::out_of_range) {
                        throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
                    }
                    // Calculate the transformation matrix (in tensor format)
                    new_kf->computeTransformTensors();

                    // Then add the keyframe into the scene
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    //Increase the keyframe's used time???
                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // Features
                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    // Pass the vector including the pixel and local coordinates to the member variables
                    // kps_pixel_ and kps_point_local_ of the new keyframe new_kf
                    // This is to avoid copying and improve efficiency
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                }
            }

            // Prepare multi resolution images for training (for Gausisan pyramid)
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                    }
                }
            }

            // Prepare for training (basic setup for training)
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                // Get the edge of scene from the point cloud
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                // Create the Gaussian model based on the cached_point_cloud_ and the scene edge
                gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                std::unique_lock<std::mutex> lock(mutex_settings_);
                // Setup the parameters for training using vector
                gaussians_->trainingSetup(opt_params_);
            }

            // Invoke training once(1st iteration)
            trainForOneIteration();

            // Finish initial mapping loop only after the map initialization and one iteration of training
            initial_mapped_ = true;
            break;
        }
        else if (pSLAM_->isShutDown()) {
            break;
        }
        else {
            // Initial conditions not satisfied
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental gaussian mapping
    int SLAM_stop_iter = 0;
    while (!isStopped()) {
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            // If the conditions for incremental mapping is fulfilled, then do mapping
            // At the same time, add new Gaussians in the Gaussian map
            combineMappingOperations();
            if (cull_keyframes_)
                // Clear unnecessary keyframes in the scene.
                cullKeyframes();
        }

        // Invoke training once
        trainForOneIteration();
        // Here we continue the training until the maximum training iteration is reached

        if (pSLAM_->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }

    // Third loop: Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}
```
{% endcode %}

`void trainColmap()` is for colmap training example only. Read the point cloud, train 3D Gaussians and save, it is similar to the run function.

### `void trainForOneIteration()`&#x20;

`void trainForOneIteration()` is the main training code, involves iterations management, Gaussian pyramid, rendering and calculating loss, saving and logging.

<pre data-full-width="true"><code>void GaussianMapper::trainForOneIteration()
{
    //Add 1 to the training iteration
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera of keyframe
    std::shared_ptr&#x3C;GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }

    writeKeyframeUsedTimes(result_dir_ / "used_times");

    // if (isdoingInactiveGeoDensify() &#x26;&#x26; !viewpoint_cam->done_inactive_geo_densify_)
    //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_.cuda();
        mask = undistort_mask_[viewpoint_cam->camera_id_];
    }
    else {
<strong>        //Get the width and height of images, original images and mask (for undistortion???)
</strong>        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    // Mutex lock for usage of the gaussian model
    std::unique_lock&#x3C;std::mutex> lock_render(mutex_render_);

    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 &#x26;&#x26; default_sh_ &#x3C; model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

    // Update learning rate
    if (pSLAM_) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times &#x3C;= opt_params_.position_lr_max_steps_ ? used_times : opt_params_.position_lr_max_steps_);
        float position_lr = gaussians_->updateLearningRate(step);
        setPositionLearningRateInit(position_lr);
    }
    else {
        gaussians_->updateLearningRate(getIteration());
    }

    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render. The returned render_pkg is a tuple, including
    // rendered images, points in the view space, visibility filter and radius.
    auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto rendered_image = std::get&#x3C;0>(render_pkg);//Rendered image
    auto viewspace_point_tensor = std::get&#x3C;1>(render_pkg);
    auto visibility_filter = std::get&#x3C;2>(render_pkg);
    auto radii = std::get&#x3C;3>(render_pkg);

<strong>    // Get rid of black edges caused by undistortion
</strong>    torch::Tensor masked_image = rendered_image * mask;

    // Loss: Calculate loss and do backpropagation
    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    float lambda_dssim = lambdaDssim();
    auto loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));
    loss.backward();

    torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;//Stop gradient calculation
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        if (keyframe_record_interval_ &#x26;&#x26;
            getIteration() % keyframe_record_interval_ == 0)
            // Record the rendered image, ground truth image, keyfrae ID, camera to the saved path
            recordKeyframeRendered(masked_image, gt_image, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_);

        // Densification
        if (getIteration() &#x3C; opt_params_.densify_until_iter_) {
            // Keep track of max radii in image-space for pruning
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
                            radii.index({visibility_filter})));
            // if (!isdoingGausPyramidTraining() || training_level &#x3C; num_gaus_pyramid_sub_levels_)
                gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            if ((getIteration() > opt_params_.densify_from_iter_) &#x26;&#x26;
                (getIteration() % densifyInterval()== 0)) {
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 20 : 0;
<strong>                // Densifify and pruning
</strong>                gaussians_->densifyAndPrune(
                    densifyGradThreshold(),
                    densify_min_opacity_,//0.005,//
                    scene_->cameras_extent_,
                    size_threshold
                );
            }

            if (opacityResetInterval()
                &#x26;&#x26; (getIteration() % opacityResetInterval() == 0
                    ||(model_params_.white_background_ &#x26;&#x26; getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();
        }

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast&#x3C;std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        // Log and save
        if (training_report_interval_ &#x26;&#x26; (getIteration() % training_report_interval_ == 0))
            GaussianTrainer::trainingReport(
                getIteration(),
                opt_params_.iterations_,
                Ll1,
                loss,
                ema_loss_for_log_,
                loss_utils::l1_loss,
                iter_time,
                *gaussians_,
                *scene_,
                pipe_params_,
                background_
            );
        if ((all_keyframes_record_interval_ &#x26;&#x26; getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");
        }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step
        // If the iteration is smaller than the maximum iteration amount, continue to optimize
        if (getIteration() &#x3C; opt_params_.iterations_) {
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);
        }
    }
}
</code></pre>
