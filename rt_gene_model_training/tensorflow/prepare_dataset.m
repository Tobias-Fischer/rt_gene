%%
%% Prepare h5 files (train+test) for PRL Gaze dataset (two eye version)
%% @Tobias Fischer (t.fischer@imperial.ac.uk), Hyung Jin Chang (hj.chang@imperial.ac.uk)

clear;
clc;

img_width = 60;
img_height = 36;
img_face_size = 224;

augmented = 1;
with_faces = 0;

% please change load_path and save_path, make sure that save_path exists!
load_path = '/recordings_hdd/';
save_path = '/recordings_hdd/mtcnn_twoeyes_inpainted_eccv/';

subfolder = '/inpainted/';
combined_file_name = '../../label_combined.txt';

person_load_path_name = {...
    [load_path,'s000_glasses',subfolder],...
    [load_path,'s001_glasses',subfolder],...
    [load_path,'s002_glasses',subfolder],...
    [load_path,'s003_glasses',subfolder],...
    [load_path,'s004_glasses',subfolder],...
    [load_path,'s005_glasses',subfolder],...
    [load_path,'s006_glasses',subfolder],...
    [load_path,'s007_glasses',subfolder],...
    [load_path,'s008_glasses',subfolder],...
    [load_path,'s009_glasses',subfolder],...
    [load_path,'s010_glasses',subfolder],...
    [load_path,'s011_glasses',subfolder],...
    [load_path,'s012_glasses',subfolder],...
    [load_path,'s013_glasses',subfolder],...
    [load_path,'s014_glasses',subfolder],...
    [load_path,'s015_glasses',subfolder],...
    [load_path,'s016_glasses',subfolder],...
};



person_save_path_name_train = {...
    [save_path, 'RT_GENE_train_s000.mat'],...
    [save_path, 'RT_GENE_train_s001.mat'],...
    [save_path, 'RT_GENE_train_s002.mat'],...
    [save_path, 'RT_GENE_train_s003.mat'],...
    [save_path, 'RT_GENE_train_s004.mat'],...
    [save_path, 'RT_GENE_train_s005.mat'],...
    [save_path, 'RT_GENE_train_s006.mat'],...
    [save_path, 'RT_GENE_train_s007.mat'],...
    [save_path, 'RT_GENE_train_s008.mat'],...
    [save_path, 'RT_GENE_train_s009.mat'],...
    [save_path, 'RT_GENE_train_s010.mat'],...
    [save_path, 'RT_GENE_train_s011.mat'],...
    [save_path, 'RT_GENE_train_s012.mat'],...
    [save_path, 'RT_GENE_train_s013.mat'],...
    [save_path, 'RT_GENE_train_s014.mat'],...
    [save_path, 'RT_GENE_train_s015.mat'],...
    [save_path, 'RT_GENE_train_s016.mat'],...
};



person_save_path_name_test = {...
    [save_path, 'RT_GENE_test_s000.mat'],...
    [save_path, 'RT_GENE_test_s001.mat'],...
    [save_path, 'RT_GENE_test_s002.mat'],...
    [save_path, 'RT_GENE_test_s003.mat'],...
    [save_path, 'RT_GENE_test_s004.mat'],...
    [save_path, 'RT_GENE_test_s005.mat'],...
    [save_path, 'RT_GENE_test_s006.mat'],...
    [save_path, 'RT_GENE_test_s007.mat'],...
    [save_path, 'RT_GENE_test_s008.mat'],...
    [save_path, 'RT_GENE_test_s009.mat'],...
    [save_path, 'RT_GENE_test_s010.mat'],...
    [save_path, 'RT_GENE_test_s011.mat'],...
    [save_path, 'RT_GENE_test_s012.mat'],...
    [save_path, 'RT_GENE_test_s013.mat'],...
    [save_path, 'RT_GENE_test_s014.mat'],...
    [save_path, 'RT_GENE_test_s015.mat'],...
    [save_path, 'RT_GENE_test_s016.mat'],...
};


for person_idx = 1:length(person_load_path_name)
    disp(num2str(person_idx));   
    %h = waitbar(0,'Please wait...');
    
    combined_file_path = [person_load_path_name{person_idx},combined_file_name];
    delimiterIn = ',';
    combined_file = importdata(combined_file_path, delimiterIn);
    
    %%
    data_num = size(combined_file.textdata,1);

    file_idx = zeros(data_num,1);
    head_pose = zeros(data_num,2);
    gaze = zeros(data_num,2);
    
    for i=1:data_num
        file_idx(i) = str2num(combined_file.textdata{i,1});
        head_pose(i,1) = str2num(combined_file.textdata{i,2}(2:end));
        head_pose(i,2) = str2num(combined_file.textdata{i,3}(1:end-1));
        gaze(i,1) = str2num(combined_file.textdata{i,4}(2:end));
        gaze(i,2) = str2num(combined_file.textdata{i,5}(1:end-1));
    end
    
    %%
    train=[];
    train.imagesL = zeros(img_width*img_height*3, 1, 'uint8');
    train.imagesR = zeros(img_width*img_height*3, 1, 'uint8');
    if with_faces
        train.imagesFace = zeros(img_face_size*img_face_size*3, 1, 'uint8');
    end
    train.gazes = zeros(2, 1);
    train.headposes = zeros(2, 1);
    
    test=[];
    test.imagesL = zeros(img_width*img_height*3, 1, 'uint8');
    test.imagesR = zeros(img_width*img_height*3, 1, 'uint8');
    if with_faces
        test.imagesFace = zeros(img_face_size*img_face_size*3, 1, 'uint8');
    end
    test.gazes = zeros(2, 1);
    test.headposes = zeros(2, 1);
    
    %%
    train_index = 0;
    test_index = 0;
    
    for f_idx = 1:data_num

        %waitbar(f_idx/data_num,h);
        
        if randi(10,1,1) >= 2
            left_eye_img_file_path  = [person_load_path_name{person_idx},'/left/left_',sprintf('%06d',file_idx(f_idx)),'_rgb.png'];
            right_eye_img_file_path = [person_load_path_name{person_idx},'/right/right_',sprintf('%06d',file_idx(f_idx)),'_rgb.png'];
            face_img_file_path = [person_load_path_name{person_idx},'/face/face_',sprintf('%06d',file_idx(f_idx)),'_rgb.png'];
            
            img_org_left  = imread(left_eye_img_file_path);
            img_org_right = imread(right_eye_img_file_path);
            img_org_face = imread(face_img_file_path);
            
            train_index = train_index+1;
            img_left  = reshape(img_org_left,  [img_height*img_width*3,1]);
            img_right = reshape(img_org_right, [img_height*img_width*3,1]);
            img_face = reshape(img_org_face, [img_face_size*img_face_size*3,1]);

            train.imagesL(:,train_index) = img_left;
            train.imagesR(:,train_index)  = img_right;
            if with_faces
                train.imagesface(:,train_index)  = img_face;
            end
            
            eye_gaze = gaze(f_idx,:);
            eye_theta = eye_gaze(2);
            eye_phi = eye_gaze(1);
            train.gazes(:,train_index) = [eye_theta; eye_phi];
            
            headpose = head_pose(f_idx,:);
            head_theta = headpose(2);
            head_phi = headpose(1);
            train.headposes(:,train_index) = [head_theta; head_phi];
            
            if augmented
                % Smoothing image 2/4
                train_index = train_index + 1;

                img_small_left   = imresize(img_org_left,2/4);
                img_restore_left = imresize(img_small_left,[img_height,img_width], 'bilinear');
                img_restore_left = reshape(img_restore_left, [img_height*img_width*3,1]);

                img_small_right   = imresize(img_org_right,2/4);
                img_restore_right = imresize(img_small_right,[img_height,img_width], 'bilinear');
                img_restore_right = reshape(img_restore_right, [img_height*img_width*3,1]);
                
                img_small_face   = imresize(img_org_face,2/4);
                img_restore_face = imresize(img_small_face,[img_face_size,img_face_size], 'bilinear');
                img_restore_face = reshape(img_restore_face, [img_face_size*img_face_size*3,1]);

                train.imagesL(:, train_index) = img_restore_left;
                train.imagesR(:, train_index) = img_restore_right;
                if with_faces
                    train.imagesFace(:, train_index) = img_restore_face;
                end

                train.gazes(:,train_index)     = [eye_theta; eye_phi];
                train.headposes(:,train_index) = [head_theta; head_phi];

                % Smoothing image 1/4
                train_index = train_index + 1;

                img_small_left   = imresize(img_org_left,1/4);
                img_restore_left = imresize(img_small_left,[img_height,img_width], 'bilinear');
                img_restore_left = reshape(img_restore_left, [img_height*img_width*3,1]);

                img_small_right   = imresize(img_org_right,1/4);
                img_restore_right = imresize(img_small_right,[img_height,img_width], 'bilinear');
                img_restore_right = reshape(img_restore_right, [img_height*img_width*3,1]);
                
                img_small_face   = imresize(img_org_face,1/4);
                img_restore_face = imresize(img_small_face,[img_face_size,img_face_size], 'bilinear');
                img_restore_face = reshape(img_restore_face, [img_face_size*img_face_size*3,1]);

                train.imagesL(:, train_index)  = img_restore_left;
                train.imagesR(:, train_index)  = img_restore_right;
                if with_faces
                    train.imagesFace(:, train_index) = img_restore_face;
                end

                train.gazes(:,train_index)     = [eye_theta; eye_phi];
                train.headposes(:,train_index) = [head_theta; head_phi];

                % Histogram equalised image
                train_index = train_index + 1;
                img_histeq_left = img_org_left;
                img_histeq_left(:,:,1) = histeq(img_org_left(:,:,1));
                img_histeq_left(:,:,2) = histeq(img_org_left(:,:,2));
                img_histeq_left(:,:,3) = histeq(img_org_left(:,:,3));
                img_histeq_left_reshape = reshape(img_histeq_left, [img_height*img_width*3,1]);
                train.imagesL(:, train_index) = img_histeq_left_reshape;

                img_histeq_right = img_org_right;
                img_histeq_right(:,:,1) = histeq(img_org_right(:,:,1));
                img_histeq_right(:,:,2) = histeq(img_org_right(:,:,2));
                img_histeq_right(:,:,3) = histeq(img_org_right(:,:,3));
                img_histeq_right_reshape = reshape(img_histeq_right, [img_height*img_width*3,1]);
                train.imagesR(:, train_index) = img_histeq_right_reshape;
                
                if with_faces
                    img_histeq_face = img_org_face;
                    img_histeq_face(:,:,1) = histeq(img_org_face(:,:,1));
                    img_histeq_face(:,:,2) = histeq(img_org_face(:,:,2));
                    img_histeq_face(:,:,3) = histeq(img_org_face(:,:,3));
                    img_histeq_face_reshape = reshape(img_histeq_face, [img_face_size*img_face_size*3,1]);
                    train.imagesFace(:, train_index) = img_histeq_face_reshape;
                end

                train.gazes(:,train_index) = [eye_theta; eye_phi];
                train.headposes(:,train_index) = [head_theta; head_phi];

                % Grayscale image
                train_index = train_index + 1;
                img_gray_left = rgb2gray(img_org_left);
                img_gray_left_three_channel = cat(3, img_gray_left, img_gray_left, img_gray_left);
                img_gray_left_three_channel = reshape(img_gray_left_three_channel, [img_height*img_width*3,1]);
                train.imagesL(:,train_index) = img_gray_left_three_channel;

                img_gray_right = rgb2gray(img_org_right);
                img_gray_right_three_channel = cat(3, img_gray_right, img_gray_right, img_gray_right);
                img_gray_right_three_channel = reshape(img_gray_right_three_channel, [img_height*img_width*3,1]);
                train.imagesR(:,train_index) = img_gray_right_three_channel;
                
                if with_faces
                    img_gray_face = rgb2gray(img_org_face);
                    img_gray_face_three_channel = cat(3, img_gray_face, img_gray_face, img_gray_face);
                    img_gray_face_three_channel = reshape(img_gray_face_three_channel, [img_face_size*img_face_size*3,1]);
                    train.imagesFace(:,train_index) = img_gray_face_three_channel;
                end

                train.gazes(:,train_index) = [eye_theta; eye_phi];
                train.headposes(:,train_index) = [head_theta; head_phi];

                % Cropping and resizing
                for iter = 1:10
                    train_index = train_index + 1;
                    rand_val = randi([0,5],4,1);
                    img_crop_coord = [rand_val(1), rand_val(2), img_width-rand_val(3), img_height-rand_val(4)];

                    img_cropped = imcrop(img_org_left, img_crop_coord);
                    img_cropped_left = imresize(img_cropped,[img_height,img_width], 'bilinear');
                    img_cropped_left = reshape(img_cropped_left, [img_height*img_width*3,1]);
                    train.imagesL(:, train_index) = img_cropped_left;

                    img_cropped = imcrop(img_org_right, img_crop_coord);
                    img_cropped_right = imresize(img_cropped,[img_height,img_width], 'bilinear');
                    img_cropped_right = reshape(img_cropped_right, [img_height*img_width*3,1]);
                    train.imagesR(:, train_index) = img_cropped_right;
                    
                    if with_faces
                        img_cropped = imcrop(img_org_face, img_crop_coord);
                        img_cropped_face = imresize(img_cropped,[img_face_size,img_face_size], 'bilinear');
                        img_cropped_face = reshape(img_cropped_face, [img_face_size*img_face_size*3,1]);
                        train.imagesFace(:, train_index) = img_cropped_face;
                    end

                    train.gazes(:,train_index) = [eye_theta; eye_phi];
                    train.headposes(:,train_index) = [head_theta; head_phi];
                end
            end
        else
            left_eye_img_file_path  = [person_load_path_name{person_idx},'/left/left_',sprintf('%06d',file_idx(f_idx)),'_rgb.png'];
            right_eye_img_file_path = [person_load_path_name{person_idx},'/right/right_',sprintf('%06d',file_idx(f_idx)),'_rgb.png'];
            face_img_file_path = [person_load_path_name{person_idx},'/face/face_',sprintf('%06d',file_idx(f_idx)),'_rgb.png'];

            img_org_left  = imread(left_eye_img_file_path);
            img_org_right = imread(right_eye_img_file_path);
            img_org_face = imread(face_img_file_path);

            test_index = test_index+1;
            img_left  = reshape(img_org_left,  [img_height*img_width*3,1]);
            img_right = reshape(img_org_right, [img_height*img_width*3,1]);
            img_face = reshape(img_org_face, [img_face_size*img_face_size*3,1]);

            test.imagesL(:,test_index) = img_left;
            test.imagesR(:,test_index)  = img_right;
            if with_faces
                test.imagesFace(:,test_index)  = img_face;
            end

            eye_gaze = gaze(f_idx,:);
            eye_theta = eye_gaze(2);
            eye_phi = eye_gaze(1);
            test.gazes(:,test_index) = [eye_theta; eye_phi];
            
            headpose = head_pose(f_idx,:);
            head_theta = headpose(2);
            head_phi = headpose(1);
            test.headposes(:,test_index) = [head_theta; head_phi];
        end
        
        fprintf('Subject %d / %d, %d / %d !\n', person_idx, length(person_load_path_name), f_idx, data_num);
    end

    train.imagesL = uint8(train.imagesL);
    train.imagesR = uint8(train.imagesR);
    if with_faces
        train.imagesFace = uint8(train.imagesFace);
    end
    train.gazes = single(train.gazes);
    train.headposes = single(train.headposes);
    
    test.imagesL = uint8(test.imagesL);
    test.imagesR = uint8(test.imagesR);
    if with_faces
        test.imagesFace = uint8(test.imagesFace);
    end
    test.gazes = single(test.gazes);
    test.headposes = single(test.headposes);
    
    train_savename = person_save_path_name_train{person_idx};
    save(train_savename, 'train', '-v7.3');
    test_savename = person_save_path_name_test{person_idx};
    save(test_savename, 'test', '-v7.3');
    
    clear train
    clear test
end

fprintf('done\n');

