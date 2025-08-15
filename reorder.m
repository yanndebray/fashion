function [XTrain, YTrain, XTest, YTest] = reorder(train_images,train_labels, test_images, test_labels)

% 1) Reorder to HxWxCxN and scale to [0,1] single
XTrain = permute(train_images, [2 3 1]);      % 28x28x60000
XTrain = reshape(XTrain, 28, 28, 1, []);      % 28x28x1x60000
XTrain = single(XTrain) / 255;                % single, scaled

% 2) Labels as categorical vector (length N)
YTrain = categorical(train_labels(:));        % 60000x1 categorical

% 3) (Optional) Do the same for test set
XTest = permute(test_images, [2 3 1]);
XTest = reshape(XTest, 28, 28, 1, []);
XTest = single(XTest) / 255;
YTest = categorical(test_labels(:));

end