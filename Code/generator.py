def generator_to_extract_image_features():
    Image_paths = []
    hsv = []
    rgb = []
    successful_images = []

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    image_ids = []
    batch_successful_images = []

    for idx, i in tqdm(enumerate(generator.filenames)):
        full_path = os.path.join(train_data_dir, i)

        image = cv2.imread(full_path)
        if image is not None:
            try:
                image = cv2.resize(image, (img_width, img_height))  # Resize the image
                hsv_hist = calculate_histogram(image, 'hsv')
                rgb_hist = calculate_histogram(image, 'rgb')
                hsv.append((idx, hsv_hist))
                rgb.append((idx, rgb_hist))
                Image_paths.append((idx, full_path))
                image_ids.append(idx)

                image = preprocess_input(image)
                batch_successful_images.append(image)

                if len(batch_successful_images) == batch_size:
                    batch_successful_images = np.stack(batch_successful_images, axis=0)
                    batch_extracted_features = model.predict(batch_successful_images)
                    successful_images.extend(batch_extracted_features)
                    batch_successful_images = []

            except:
                print(f"Failed processing {full_path}")
                continue

    # process remaining images
    if batch_successful_images:
        batch_successful_images = np.stack(batch_successful_images, axis=0)
        batch_extracted_features = model.predict(batch_successful_images)
        successful_images.extend(batch_extracted_features)
    
    df_hsv = pd.DataFrame(hsv, columns=["ID", "Histogram"])
    df_hsv.to_pickle('ID_hsv_3.pkl')

    df_rgb = pd.DataFrame(rgb, columns=["ID", "Histogram"])
    df_rgb.to_pickle('ID_rgb_3.pkl')

    df_image_paths = pd.DataFrame(Image_paths, columns=["ID", "Path"])
    df_image_paths.to_pickle('ID_path_3.pkl')

    df_extracted_features = pd.DataFrame({
        "ID": image_ids,
        "Embeddings": successful_images
    })
    df_extracted_features.to_pickle('ID_Embeddings_3.pkl')
  
# generator_to_extract_image_features()
