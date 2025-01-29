clc
clearvars
close all

path_img = 'ground_images_location';
imgFiles = dir(fullfile(path_img, '*.jpg')); 
no_images = numel(imgFiles); %number of known images

% data matrix
data = zeros(numel(imread(fullfile(path_img, imgFiles(1).name))), no_images);

for i = 1:no_images
    img = imread(fullfile(path_img, imgFiles(i).name));
    data(:, i) = img(:); 
end

%% apply PCA
%  data standardization 
data_mean = mean(data, 2);
data = data - data_mean;

% covariance matrix
mat_cov = data*data';

% eigenvectors and eigenvalues
[eigen_vectors, eigen_values] = eig(mat_cov);

% sort eigenvalues by descending eigenvalues
[~, idx] = sort(diag(eigen_values), 'descend');
eigen_vectors = eigen_vectors(:, idx);
	
% create feature vector from eigenvectors
no_pc = 50; %set number of principle components
selected_eigenvectors = eigen_vectors(:, 1:no_pc);

% data projection in the new subspace
data_projection = selected_eigenvectors' * data;

%% test on new images
test_set = imageDatastore('test_images_location');
for i=1:numel(test_set.Files)
    new_image = readimage(test_set,i);

    % image (data) standardization
    eigenvector_new_im = double(new_image(:)) - data_mean; %data_mean is the one previously calculated

    % proiectarea imaginii in subspatiul definit de componentele principale
    % project the new image in the subspace defined by the PC previously calculated
    new_im_projection = selected_eigenvectors' * eigenvector_new_im;

    % similarity score - euclidian distance
    similarity_score = zeros(1, no_images);

    for j = 1:no_images
        similarity_score(j) = norm(new_im_projection - data_projection(:, j));
    end

    % identificarea celei mai potrivite imagini (cea cu cea mai mica diferenta)
    % identify the appropiate image from the known ones (the smalles distance)
    [min_score, appropiate_idx] = min(similarity_score);

    % print the final result
    figure;
    subplot(1, 2, 1); imshow(new_image); title('New Image');
    subplot(1, 2, 2); imshow(uint8(reshape(data(:, appropiate_idx) + data_mean, size(new_image)))); title('Associated image');
    disp(['Similarity score: ', num2str(min_score)]);
end
