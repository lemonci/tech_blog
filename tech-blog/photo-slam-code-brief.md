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

<pre data-full-width="true"><code><strong>    // Create SLAM system. It initializes all system threads and gets ready to process frames.
</strong><strong>    // Create the pointer pSLAM to point to the SLAM system
</strong>    std::shared_ptr&#x3C;ORB_SLAM3::System> pSLAM =
        std::make_shared&#x3C;ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::RGBD);
    float imageScale = pSLAM->GetImageScale();

<strong>    // Create GaussianMapper
</strong><strong>    // Create the pointer pGuasMapper to point to the 3D Gaussian Mapper,
</strong><strong>    // the input parameter is the SLAM system, configs for 3D Gaussian 
</strong><strong>    // and output directory
</strong>    std::filesystem::path gaussian_cfg_path(argv[3]);
    std::shared_ptr&#x3C;GaussianMapper> pGausMapper =
        std::make_shared&#x3C;GaussianMapper>(
            pSLAM, gaussian_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&#x26;GaussianMapper::run, pGausMapper.get());

<strong>    // Create Gaussian Viewer
</strong>    std::thread viewer_thd;
    std::shared_ptr&#x3C;ImGuiViewer> pViewer;
<strong>    // If the GUI config is true, we start the viewer thread and
</strong><strong>    // pass SLAM system pointer and 3D Gaussian Mapper pointer as input
</strong>    if (use_viewer)
    {  
        pViewer = std::make_shared&#x3C;ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&#x26;ImGuiViewer::run, pViewer.get());
    }
</code></pre>

4. Then we output the system information.\
   Load the RGB and depth information and check if they are valid.\
   Scale RGB and depth image according to `imageScale`.\
   Timer (`chrono` is the new timing method in C++ 11)\
   Import RGB, depth image and frame time into `pSLAM` system.

<pre data-full-width="true"><code><strong>// (The exiciting) Main loop.
</strong><strong>// Load all images in sequence and deal with them.
</strong>    cv::Mat imRGB, imD;
    for (int ni = 0; ni &#x3C; nImages; ni++)
    {
        if (pSLAM->isShutDown())
            break;
<strong>        // Read image and depthmap from file
</strong>        imRGB = cv::imread(std::string(argv[4]) + "/" + vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        cv::cvtColor(imRGB, imRGB, CV_BGR2RGB);
        imD = cv::imread(std::string(argv[4]) + "/" + vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

<strong>        // Handle the empty images
</strong>        if (imRGB.empty())
        {
            std::cerr &#x3C;&#x3C; std::endl &#x3C;&#x3C; "Failed to load image at: "
                      &#x3C;&#x3C; std::string(argv[4]) &#x3C;&#x3C; "/" &#x3C;&#x3C; vstrImageFilenamesRGB[ni] &#x3C;&#x3C; std::endl;
            return 1;
        }
        if (imD.empty())
        {
            std::cerr &#x3C;&#x3C; std::endl &#x3C;&#x3C; "Failed to load depth image at: "
                      &#x3C;&#x3C; std::string(argv[4]) &#x3C;&#x3C; "/" &#x3C;&#x3C; vstrImageFilenamesD[ni] &#x3C;&#x3C; std::endl;
            return 1;
        }

<strong>        // Scale the images
</strong>        if (imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

<strong>        // Timing
</strong>        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

<strong>        // Pass the image to the SLAM system
</strong>        pSLAM->TrackRGBD(imRGB, imD, tframe, std::vector&#x3C;ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni]);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast&#x3C;std::chrono::duration&#x3C;double>>(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

<strong>        // Wait to load the next frame
</strong>        double T = 0;
        if (ni &#x3C; nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack &#x3C; T)
            usleep((T - ttrack) * 1e6);
    }
</code></pre>

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

<pre data-full-width="true"><code><strong>// Constructor
</strong>GaussianMapper::GaussianMapper(
    std::shared_ptr&#x3C;ORB_SLAM3::System> pSLAM,
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
<strong>    // In standard C++ library, this code is used to set up the random seed.
</strong>      std::srand(seed);
<strong>    // This is the random seed set up for PyTorch/litorch.
</strong>    torch::manual_seed(seed);

<strong>    // Device(set up according to the input device)
</strong>    if (device_type == torch::kCUDA &#x26;&#x26; torch::cuda::is_available()) {
        std::cout &#x3C;&#x3C; "[Gaussian Mapper]CUDA available! Training on GPU." &#x3C;&#x3C; std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cout &#x3C;&#x3C; "[Gaussian Mapper]Training on CPU." &#x3C;&#x3C; std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

<strong>    // Create the output directory for results
</strong>    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

<strong>    // Read the config file
</strong>    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

<strong>    // Set up the background color
</strong>    std::vector&#x3C;float> bg_color;
    if (model_params_.white_background_) //If it is white
        bg_color = {1.0f, 1.0f, 1.0f};
    else //Otherwise it is black
        bg_color = {0.0f, 0.0f, 0.0f};
<strong>    // Convert bg_color to a tensor and save it in the variable named background_
</strong><strong>    // Here we use torch::TensorOptions() to specify the options for the tensor
</strong><strong>    // including the data type to torch::kFloat32 and the device type
</strong>    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

<strong>    // Initialize scene and model including the SH parameter 
</strong><strong>    // Initializ 3DGS tensor
</strong>    gaussians_ = std::make_shared&#x3C;GaussianModel>(model_params_);
    scene_ = std::make_shared&#x3C;GaussianScene>(model_params_);

<strong>    // Mode (If there is no SLAM results then return)
</strong>    if (!pSLAM) {
        // NO SLAM
        return;
    }

<strong>    // Specify tensor types
</strong>    switch (pSLAM->getSensorType())
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

<strong>    // Cameras
</strong>    // TODO: not only monocular
    auto settings = pSLAM->getSettings();
    cv::Size SLAM_im_size = settings->newImSize();

<strong>    //Get the undistortion parameters of the camera
</strong>    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );

<strong>    //Get all cameras and the intrinsics accordingly in a loop
</strong>    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    for (auto&#x26; SLAM_camera : vpCameras) {
        Camera camera;//Create a camera class

<strong>        //Get the id of the camera
</strong>        camera.camera_id_ = SLAM_camera->GetId();

<strong>        //Get the type of the camera to fetch the instrinc matrix
</strong>        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            // Old K, i.e. K in SLAM
            cv::Mat K = (
                cv::Mat_&#x3C;float>(3, 3)
                    &#x3C;&#x3C; SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );

            // camera.width_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.width
            //                                              : graphics_utils::roundToIntegerMultipleOf16(
            //                                                    undistort_params.old_size_.width);
            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast&#x3C;float>(camera.width_) / undistort_params.old_size_.width;

            // camera.height_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.height
            //                                               : graphics_utils::roundToIntegerMultipleOf16(
            //                                                     undistort_params.old_size_.height);
            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast&#x3C;float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l &#x3C; num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
            camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
            camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
            camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_&#x3C;float>(3, 3)
                    &#x3C;&#x3C; camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

<strong>            // Undistortion
</strong>            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
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
                    this->stereo_Q_.at&#x3C;float>(0, 0) = 1.0f;
                    this->stereo_Q_.at&#x3C;float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at&#x3C;float>(1, 1) = 1.0f;
                    this->stereo_Q_.at&#x3C;float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at&#x3C;float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at&#x3C;float>(3, 2) = 1.0f / stereo_baseline_length_;
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
</code></pre>

### `void run()`&#x20;

&#x20;`void run()` is the main main process, read the camera pose and point cloud, prepare the images of multiple resolutions and call the train function `trainForOneIteration()` .

<pre data-full-width="true"><code>void GaussianMapper::run()
{
<strong>    // First loop: Initial gaussian mapping
</strong><strong>    // Initially, we use a loop to initialize the 3D Gaussian mapping process and single iteration training
</strong><strong>    // The loop will end once the initialization is executed and the first training iteration is finished.
</strong>    while (!isStopped()) {
<strong>        // Check conditions for initial mapping
</strong><strong>        // We need ORBSLAM still running and the number of keyframes is over a certain amount
</strong>        if (hasMetInitialMappingConditions()) {

<strong>            //Clean the mapping operation of orbslam3
</strong>            pSLAM_->getAtlas()->clearMappingOperation();

<strong>            // Get initial sparse map from orbSLAM's current map
</strong>            auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
            std::vector&#x3C;ORB_SLAM3::KeyFrame*> vpKFs;
            std::vector&#x3C;ORB_SLAM3::MapPoint*> vpMPs;
            {
<strong>                // Lock the map of orb3, basically pause the tracking during mapping
</strong>                std::unique_lock&#x3C;std::mutex> lock_map(pMap->mMutexMapUpdate);
<strong>                //Get keyframes
</strong>                vpKFs = pMap->GetAllKeyFrames();
<strong>                //Get map points
</strong>                vpMPs = pMap->GetAllMapPoints();

<strong>                //Iterate all map points to get their 3D coordinates and colors. Then place them into scene_
</strong>                for (const auto&#x26; pMP : vpMPs){
                    Point3D point3D;
                    auto pos = pMP->GetWorldPos();
                    point3D.xyz_(0) = pos.x();
                    point3D.xyz_(1) = pos.y();
                    point3D.xyz_(2) = pos.z();
                    auto color = pMP->GetColorRGB();
                    point3D.color_(0) = color(0);
                    point3D.color_(1) = color(1);
                    point3D.color_(2) = color(2);

<strong>                    //Place the points and colors into cached_point_cloud_
</strong>                    scene_->cachePoint3D(pMP->mnId, point3D);
                }

<strong>                //Then iterate all keyframes
</strong>                for (const auto&#x26; pKF : vpKFs){
                    //Create a new Gaussian keyframe based on the keyframe id and current iteration number
                    std::shared_ptr&#x3C;GaussianKeyframe> new_kf = std::make_shared&#x3C;GaussianKeyframe>(pKF->mnId, getIteration());
                    //The closet and farest depth
                    new_kf->zfar_ = z_far_;
                    new_kf->znear_ = z_near_;

<strong>                    //Get pose of the according keyframe
</strong>                    auto pose = pKF->GetPose();
                    //Set pose
                    new_kf->setPose(
                        pose.unit_quaternion().cast&#x3C;double>(),
                        pose.translation().cast&#x3C;double>());

<strong>                    // Get image information
</strong>                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    try {
                        // Camera
                        Camera&#x26; camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                        new_kf->setCameraParams(camera);

                        // Image (left if STEREO)
                        cv::Mat imgRGB = pKF->imgLeftRGB;
                        if (this->sensor_type_ == STEREO)
                            imgRGB_undistorted = imgRGB;
                        else
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
<strong>                        // Auxiliary Image ???
</strong>                        cv::Mat imgAux = pKF->imgAuxiliary;
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
<strong>                    // Calculate the transformation matrix (in tensor format)
</strong>                    new_kf->computeTransformTensors();

<strong>                    // Then add the keyframe into the scene
</strong>                    scene_->addKeyframe(new_kf, &#x26;kfid_shuffled_);

<strong>                    //Increase the keyframe's used time ???
</strong>                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

<strong>                    // Features ???
</strong>                    std::vector&#x3C;float> pixels;
                    std::vector&#x3C;float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
<strong>                    // Pass the vector including the pixel and local coordinates to the member variables
</strong><strong>                    // kps_pixel_ and kps_point_local_ of the new keyframe new_kf
</strong><strong>                    // This is to avoid copying and improve efficiency
</strong>                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                }
            }

<strong>            // Prepare multi resolution images for training (for Gausisan pyramid)
</strong>            for (auto&#x26; kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l &#x3C; num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l &#x3C; num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                    }
                }
            }

<strong>            // Prepare for training (basic setup for training)
</strong>            {
                std::unique_lock&#x3C;std::mutex> lock_render(mutex_render_);
                // Get the edge of scene from the point cloud
                scene_->cameras_extent_ = std::get&#x3C;1>(scene_->getNerfppNorm());
                // Create the Gaussian model based on the cached_point_cloud_ and the scene edge
                gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                std::unique_lock&#x3C;std::mutex> lock(mutex_settings_);
                // Setup the parameters for training using vector
                gaussians_->trainingSetup(opt_params_);
            }

<strong>            // Invoke training once(1st iteration)
</strong>            trainForOneIteration();

<strong>            // Finish initial mapping loop only after the map initialization and one iteration of training
</strong>            initial_mapped_ = true;
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

<strong>    // Second loop: Incremental gaussian mapping
</strong>    int SLAM_stop_iter = 0;
    while (!isStopped()) {
<strong>        // Check conditions for incremental mapping
</strong>        if (hasMetIncrementalMappingConditions()) {
            // If the conditions for incremental mapping is fulfilled, then do mapping
            // At the same time, add new Gaussians in the Gaussian map
            combineMappingOperations();
            if (cull_keyframes_)
                // Clear unnecessary keyframes in the scene.
                cullKeyframes();
        }

<strong>        // Invoke training once
</strong>        trainForOneIteration();
<strong>        // Here we continue the training until the maximum training iteration is reached
</strong>
        if (pSLAM_->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }

<strong>    // Third loop: Tail gaussian optimization
</strong>    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() - SLAM_stop_iter &#x3C;= n_delay_iters || getIteration() % densify_interval &#x3C;= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

<strong>    // Save and clear
</strong>    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}
</code></pre>

### `void trainForOneIteration()`&#x20;

`void trainForOneIteration()` is the main training code, involves iterations management, Gaussian pyramid, rendering and calculating loss, saving and logging.

<pre data-full-width="true"><code>void GaussianMapper::trainForOneIteration()
{
<strong>    //Add 1 to the training iteration
</strong>    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

<strong>    // Pick a random Camera of keyframe
</strong>    std::shared_ptr&#x3C;GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
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

<strong>    // Mutex lock for usage of the gaussian model
</strong>    std::unique_lock&#x3C;std::mutex> lock_render(mutex_render_);

<strong>    // Every 1000 iterations, we increase the levels of SH up to a maximum degree
</strong>    if (getIteration() % 1000 == 0 &#x26;&#x26; default_sh_ &#x3C; model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

<strong>    // Update learning rate
</strong>    if (pSLAM_) {
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

<strong>    // Render. The returned render_pkg is a tuple, including
</strong><strong>    // rendered images, points in the view space, visibility filter and radius.
</strong>    auto render_pkg = GaussianRenderer::render(
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

<strong>    // Loss: Calculate loss and do backpropagation
</strong>    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
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

<strong>        // Densification
</strong>        if (getIteration() &#x3C; opt_params_.densify_until_iter_) {
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

<strong>        // Log and save
</strong>        if (training_report_interval_ &#x26;&#x26; (getIteration() % training_report_interval_ == 0))
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

<strong>        // Optimizer step
</strong><strong>        // If the iteration is smaller than the maximum iteration amount, continue to optimize
</strong>        if (getIteration() &#x3C; opt_params_.iterations_) {
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);
        }
    }
</code></pre>

### `void combineMappingOperations()`

`void combineMappingOperations()`  is to combine the mappers, using the functions of orb-slam3 to do BA and loop closure.

<pre data-full-width="true"><code>void GaussianMapper::combineMappingOperations()
{
<strong>    // Get Mapping Operations
</strong>    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
<strong>        // Handle local bundle adjustment
</strong>        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            // std::cout &#x3C;&#x3C; "[Gaussian Mapper]Local BA Detected."
            //           &#x3C;&#x3C; std::endl;

<strong>            // Get new keyframes
</strong>            auto&#x26; associated_kfs = opr.associatedKeyFrames();

<strong>            // Add keyframes to the scene
</strong>            for (auto&#x26; kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get&#x3C;0>(kf);
                std::shared_ptr&#x3C;GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
<strong>                // If the keyframe is already in the scene, only update the pose.
</strong><strong>                // Otherwise create a new one
</strong>                if (pkf) {
                    auto&#x26; pose = std::get&#x3C;2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast&#x3C;double>(),
                        pose.translation().cast&#x3C;double>());
                    pkf->computeTransformTensors();

                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                }
                else {
                    handleNewKeyframe(kf);
                }
            }

<strong>            // Get new points
</strong><strong>            // Theoratically we can get the new points by removing the cooresponding points ???
</strong>            auto&#x26; associated_points = opr.associatedMapPoints();
            auto&#x26; points = std::get&#x3C;0>(associated_points);
            auto&#x26; colors = std::get&#x3C;1>(associated_points);

<strong>            // Add new points to the model
</strong>            if (initial_mapped_ &#x26;&#x26; points.size() >= 30) {
<strong>            // No gradient calculation below
</strong>                torch::NoGradGuard no_grad;
                std::unique_lock&#x3C;std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }
        }
        break;

<strong>        // Loop closure bundle adjustment
</strong>        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cout &#x3C;&#x3C; "[Gaussian Mapper]Loop Closure Detected."
                      &#x3C;&#x3C; std::endl;

<strong>            // Get the loop keyframe scale modification factor
</strong>            float loop_kf_scale = opr.mfScale;

<strong>            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
</strong>            auto&#x26; associated_kfs = opr.associatedKeyFrames();
<strong>            // Mark the transformed points to avoid transforming more than once
</strong>            torch::Tensor point_not_transformed_flags =
                torch::full(
                    {gaussians_->xyz_.size(0)},
                    true,
                    torch::TensorOptions().device(device_type_).dtype(torch::kBool));
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
            int num_transformed = 0;
<strong>            // Add keyframes to the scene
</strong>            for (auto&#x26; kf : associated_kfs) {
<strong>                // Keyframe Id
</strong>                auto kfid = std::get&#x3C;0>(kf);
                std::shared_ptr&#x3C;GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
<strong>                // In case new points are added in handleNewKeyframe()
</strong>                int64_t num_new_points = gaussians_->xyz_.size(0) - point_not_transformed_flags.size(0);
                if (num_new_points > 0)
                    point_not_transformed_flags = torch::cat({
                        point_not_transformed_flags,
                        torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                        /*dim=*/0);
<strong>                // If kf is already in the scene, evaluate the change in pose,
</strong><strong>                // if too large we perform loop correction on its visible model points.
</strong><strong>                // If not in the scene, create a new one.
</strong>                if (pkf) {
                    auto&#x26; pose = std::get&#x3C;2>(kf);
<strong>                    // If is loop closure kf
</strong>// if (std::get&#x3C;4>(kf)) {
// renderAndRecordKeyframe(pkf, result_dir_, "_0_before_loop_correction");
                        Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                        Sophus::SE3f inv_pose = pose.inverse();
                        Sophus::SE3f diff_pose = inv_pose * original_pose;
                        bool large_rot = !diff_pose.rotationMatrix().isApprox(
                            Eigen::Matrix3f::Identity(), large_rot_th_);
                        bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                            1.0, large_trans_th_);
                        if (large_rot || large_trans) {
                            std::cout &#x3C;&#x3C; "[Gaussian Mapper]Large loop correction detected, transforming visible points of kf "
                                    &#x3C;&#x3C; kfid &#x3C;&#x3C; std::endl;
                            diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                            diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                            diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                            torch::Tensor diff_pose_tensor =
                                tensor_utils::EigenMatrix2TorchTensor(
                                    diff_pose.matrix(), device_type_).transpose(0, 1);
                            {
                                std::unique_lock&#x3C;std::mutex> lock_render(mutex_render_);
                                gaussians_->scaledTransformVisiblePointsOfKeyframe(
                                    point_not_transformed_flags,
                                    diff_pose_tensor,
                                    pkf->world_view_transform_,
                                    pkf->full_proj_transform_,
                                    pkf->creation_iter_,
                                    stableNumIterExistence(),
                                    num_transformed,
                                    loop_kf_scale); // selected xyz *= s
                            }
<strong>                            // Give loop keyframes times of use
</strong>                            increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
// renderAndRecordKeyframe(pkf, result_dir_, "_1_after_loop_transforming_points");
// std::cout&#x3C;&#x3C;num_transformed&#x3C;&#x3C;std::endl;
                        }
// }
                    pkf->setPose(
                        pose.unit_quaternion().cast&#x3C;double>(),
                        pose.translation().cast&#x3C;double>());
                    pkf->computeTransformTensors();
// if (std::get&#x3C;4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");
                }
                else {
                    handleNewKeyframe(kf);
                }
            }
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
// keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));

<strong>            // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
</strong>            auto&#x26; associated_points = opr.associatedMapPoints();
            auto&#x26; points = std::get&#x3C;0>(associated_points);
            auto&#x26; colors = std::get&#x3C;1>(associated_points);

<strong>            // Add new points to the model
</strong>            if (initial_mapped_ &#x26;&#x26; points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock&#x3C;std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }

<strong>            // Mark this iteration
</strong>            loop_closure_iteration_ = true;
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cout &#x3C;&#x3C; "[Gaussian Mapper]Scale refinement Detected. Transforming all kfs and points..."
                      &#x3C;&#x3C; std::endl;

            float s = opr.mfScale;
            Sophus::SE3f&#x26; T = opr.mT;
            if (initial_mapped_) {
                // Apply the scaled transformation on gaussian model points
                {
                    std::unique_lock&#x3C;std::mutex> lock_render(mutex_render_);
                    gaussians_->applyScaledTransformation(s, T);
                }
                // Apply the scaled transformation to the scene
                scene_->applyScaledTransformation(s, T);
            }
            else { // TODO: the workflow should not come here, delete this branch
                // Apply the scaled transformation to the cached points
                for (auto&#x26; pt : scene_->cached_point_cloud_) {
                    // pt &#x3C;- (s * Ryw * pt + tyw)
                    auto&#x26; pt_xyz = pt.second.xyz_;
                    pt_xyz *= s;
                    pt_xyz = T.cast&#x3C;double>() * pt_xyz;
                }

<strong>                // Apply the scaled transformation on gaussian keyframes
</strong>                for (auto&#x26; kfit : scene_->keyframes()) {
                    std::shared_ptr&#x3C;GaussianKeyframe> pkf = kfit.second;
                    Sophus::SE3f Twc = pkf->getPosef().inverse();
                    Twc.translation() *= s;
                    Sophus::SE3f Tyc = T * Twc;
                    Sophus::SE3f Tcy = Tyc.inverse();
                    pkf->setPose(Tcy.unit_quaternion().cast&#x3C;double>(), Tcy.translation().cast&#x3C;double>());
                    pkf->computeTransformTensors();
                }
            }
        }
        break;

        default:
        {
            throw std::runtime_error("MappingOperation type not supported!");
        }
        break;
        }
    }
}
</code></pre>

### `void GaussianMapper::handleNewKeyframe`

`void GaussianMapper::handleNewKeyframe`  sets up the pose, camera, image, auxiliary image of the new keyframe; puts the new keyframe into the scene. It gives the used time of the keyframe and put it into the training sliding window,&#x20;

<pre><code><strong>void GaussianMapper::handleNewKeyframe(
</strong>    std::tuple&#x3C; unsigned long/*Id*/,
                unsigned long/*CameraId*/,
                Sophus::SE3f/*pose*/,
                cv::Mat/*image*/,
                bool/*isLoopClosure*/,
                cv::Mat/*auxiliaryImage*/,
                std::vector&#x3C;float>,
                std::vector&#x3C;float>,
                std::string> &#x26;kf)
{
    std::shared_ptr&#x3C;GaussianKeyframe> pkf =
        std::make_shared&#x3C;GaussianKeyframe>(std::get&#x3C;0>(kf), getIteration());
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
<strong>    // Pose
</strong>    auto&#x26; pose = std::get&#x3C;2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast&#x3C;double>(),
        pose.translation().cast&#x3C;double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    try {
<strong>        // Camera
</strong>        Camera&#x26; camera = scene_->cameras_.at(std::get&#x3C;1>(kf));
        pkf->setCameraParams(camera);

<strong>        // Image (left if STEREO)
</strong>        cv::Mat imgRGB = std::get&#x3C;3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
<strong>        // Auxiliary Image
</strong>        cv::Mat imgAux = std::get&#x3C;5>(kf);
        if (this->sensor_type_ == RGBD)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get&#x3C;8>(kf);
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
<strong>    // Add the new keyframe to the scene
</strong>    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &#x26;kfid_shuffled_);

<strong>    // Give new keyframes times of use and add it to the training sliding window
</strong>    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

<strong>    // Get dense point cloud from the new keyframe to accelerate training
</strong>    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;
    pkf->kps_pixel_ = std::move(std::get&#x3C;6>(kf));
    pkf->kps_point_local_ = std::move(std::get&#x3C;7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

<strong>    // Prepare multi resolution images for pyramid training
</strong>    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l &#x3C; num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l &#x3C; num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }
}
</code></pre>

### Other functions

#### `bool GaussianMapper::isStopped()`

`bool GaussianMapper::isStopped()` returns the private variable `stopped_` of the class object`GaussianMapper`, at the same time it use the mutual lock to ensure the visit to this variable is thread-safe in the multi-thread environment.

<pre data-full-width="true"><code>bool GaussianMapper::isStopped()
{
<strong>    // Create a exclusive lock lock_status, also lockes the mutex mutex_status_
</strong><strong>    // The purpose of the exclusive lock is to ensure during the processing time of the function,
</strong><strong>    // other threads cannot visit the protected data
</strong>    std::unique_lock&#x3C;std::mutex> lock_status(this->mutex_status_);
<strong>    // The following line returns the private variable stopped_ of the class object GaussianMapper
</strong><strong>    // Because we have got the exclusive access to mutex_status_, we can safely read the value of this member variable.
</strong>    return this->stopped_;

    // When the function finishes and leave the scpe, the exclusive lock lock_status 
    // will release automatically and release mutex_status_
}
</code></pre>

#### `void trainColmap()`&#x20;

`void trainColmap()` is for colmap training example only. Read the point cloud, train 3D Gaussians and save, it is similar to the run function.

#### Renderings

There are mainly three rendering functions to render rgb, depth, loss and to output the evaluation metrics: `void recordKeyframeRendered(/*param*/)`, `void renderAndRecordKeyframe(/*param*/)` and `void renderAndRecordAllKeyframes(/*param*/)`.

#### Loader

`void loadPly(/*param*/)` not only loads the point cloud, but also load the camera intrinsics to undistort the images and reset the size of the image.

#### Densification based on inactive  points

`void increasePcdByKeyframeInactiveGeoDensify(/*param*/)` densifies the point cloud based on inactive points according to different types of sensors.

## src/[gaussian\_model.cpp](https://github.com/KwanWaiPang/Photo-SLAM_comment/blob/main/src/gaussian_model.cpp)

### Head File

This is a main class. According to the head file, the main member variables include:

`torch::DeviceType device_type_`: type of device

The current degree and maximum degred of sperical harmonic: `int active_sh_degree_` and `int max_sh_degree_` .

The parameters of 3D Gaussians: \
`torch::Tensor xyz_`: position\
`torch::Tensor features_dc_`: direct component, aka, the inherent color of the Gaussian when SH=0\
`torch::Tensor features_rest_`: remaining components, the remaining spherical harmonics coefficients for SH≥1.\
`torch:: Tensor opacity_`: opacity alpha\
`torch::Tensor scaling_` : scaling factors\
`torch::Tensor rotation_` : rotation factors\
`torch::Tensor xyz_gradient_accum_` : accumulated gradient in the Gaussians\
`torch::Tensor denom_` : denominator tensor used for computing the average gradient accumulation for each Gaussian point during densification\
`torch::Tensor exist_since_iter_` : The iteration when this Gaussian is added to the map

Optimizer: `std::shared_ptr <torch::optim::Adam> optimizer_`&#x20;

Position and color of the sparse point cloud: `torch::Tensor sparse_points_xyz` and `torch::Tensor sparse_points_color` .

Two macros:\
`# define GAUSSIAN_MODEL_TENSORS_TO_VEC` : to store the group of tensors into a vector (data type: `std::vector<torch::Tensor>`)

```
#define GAUSSIAN_MODEL_TENSORS_TO_VEC                        \
    this->Tensor_vec_xyz_ = {this->xyz_};                    \
    this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
    this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
    this->Tensor_vec_opacity_ = {this->opacity_};            \
    this->Tensor_vec_scaling_ = {this->scaling_};            \
    this->Tensor_vec_rotation_ = {this->rotation_};
```

\
`#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)` : to initialize multiple tensors and place them to the specific device. This macro accpets one parameter `device_type` for specific device.

```
#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                             \
    this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));                \
    this->features_dc_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->features_rest_ = torch::empty(0, torch::TensorOptions().device(device_type));      \
    this->scaling_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->rotation_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    this->opacity_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->max_radii2D_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->xyz_gradient_accum_ = torch::empty(0, torch::TensorOptions().device(device_type)); \
    this->denom_ = torch::empty(0, torch::TensorOptions().device(device_type));              \
    GAUSSIAN_MODEL_TENSORS_TO_VEC 
```

The constructor `GaussianModel::GaussianModel()`. The input is sh\_degree and model parameters.

`torch::Tensor getCovarianceActivation(int scaling_modifier = 1)` : Calculate the covariance matrix from the scaling and rotation, at the same time outputs symmetric uncertainty (NICE!)

`void createFromPcd`: Initialize the Gaussians from the point clouds. Create, fill and fuse the point cloud tensor and color tensor, deal with the color with SH degree. Calculate other properties of the point cloud.

`void increasePcd`: Add the new point cloud data (with color) into the existing Gaussian model.

`void trainingSetup(/*param*/)`: set up the paramters of adam optimizer and learning rate.

`float updateLearningRate(int step)`: Update the learning rate based on the step, call the pre-defined  `exponLrFunc` function to get a continous leanring rate decrease function.

The following functions handle the densification, clone, pruning and split of Gaussians:\
`void densifyAndSplit`, `void densifyAndClone`, `void densifyAndPrune` .

`void densificationPostfix()` extend s the newly created Gaussians (already densified) into the existing Gaussian model.

`void scaledTransformVisiblePointsOfKeyframe()` prepares and marks the model's Gaussian 3D points that are visible from a given keyframe, applying a uniform scale and view/projection transforms, then registers the transformed point and rotation tensors for optimization.

## `src/`[`gaussian_scene.cpp`](https://github.com/KwanWaiPang/Photo-SLAM_comment/blob/main/src/gaussian_scene.cpp)

### Head File

`GaussianScene::GaussianScene(/*param*/)` is the constructor. It can read the Gaussian scene already trained.

`void GaussianScene::addCamera(/*param*/)` contains a bunch of get/set methods, to get the camera, keyframes and 3D points\`.

`void GaussianScne::applyScaledTransformation(/*param*/)`  applies a uniform scale to every keyframe's translation and then applies a rigid transform to each keyframe pose. It updates each GaussianKeyframe's stored pose and recomputes its transform tensors.\
The net effect is a similarity transform (scale + rigid transform) applied to all keyframe positions and poses. Rotation comes only from the rigid transformation and the original pose; the scale is applied only to translations.

## Utility functions for Gaussians

### `src/`[`gaussian_trainer.cpp`](https://github.com/KwanWaiPang/Photo-SLAM_comment/blob/main/src/gaussian_trainer.cpp)

`void GaussianTrainer:trainingOnce()`: Train the Gaussians once, intialize the iterations, read the training options, set up the background color

`void GaussianTrainer::trainingReport()`: Input and Output

### `src/`[`gaussian_keyframe.cpp`](https://github.com/KwanWaiPang/Photo-SLAM_comment/blob/main/src/gaussian_keyframe.cpp)

We definitely shall not forget our favorite keyframing part of a SLAM system :)

The headfile mainly defines the camera id, parameters, size of the images and gaussian pyramid, as well as the original image. The main functions in the cpp file includes:

`GaussianKeyframe::getProjectionMatrix()`: to get the projection matrix

`GaussianKeyframe::getWorld2View2()`: the transformation matrix from world to camera frame

`void computeTransformTensors()`:  Use the two functions above to calculate the frame trasformation: world frame -> camera frame -> pixel frame.

`int getCurretGausPyramidLevel()`: get the current layer of the Gaussian pyramid

### `src/`[`gaussian_rasterizer.cpp`](https://github.com/KwanWaiPang/Photo-SLAM_comment/blob/main/src/gaussian_rasterizer.cpp)

`gaussian_rasterizer` mainly includes

`GaussianRasterizer::markVisibleGaussians` : the selection of visibile Gaussians

`GaussianRasterizerFunction::forward`: forwarding&#x20;

`GaussianRasterizerFunction::backward` : backword propagation

### `src/`[`gaussian_renderer.cpp`](https://github.com/KwanWaiPang/Photo-SLAM_comment/blob/main/src/gaussian_renderer.cpp)

It only has one function `GaussianRenderer::render`, as a wrapper of the forward function above but it handles the calculation of covariance and spherical harmonic degree. It is similar to the [`gaussian_render/__init__.py`](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py).

