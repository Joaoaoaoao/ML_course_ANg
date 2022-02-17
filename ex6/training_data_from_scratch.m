%% ingests all 1400 spam emails

spam_emails = dir('./spam_2/0*');
ham_emails = dir('./hard_ham/0*');

X_1 = zeros(length(spam_emails),1899);
y_1 = ones(length(spam_emails),1);

X_0 = zeros(length(ham_emails),1899);
y_0 = zeros(length(ham_emails),1);


% spam

for n = 1:length(spam_emails)
    f = ['./spam_2/' spam_emails(n).name ];
    file_contents = readFile(f);
    word_indices  = processEmail(file_contents);
    X_1(n,:) = emailFeatures(word_indices);
end

% ham

for n = 1:length(ham_emails)
    f = ['./spam_2/' spam_emails(n).name ];
    file_contents = readFile(f);
    word_indices  = processEmail(file_contents);
    X_0(n,:) = emailFeatures(word_indices);

end

X_scratch =[X_0; X_1];
y_scratch =[y_0; y_1];

save('train_scrach','X_scratch', 'y_scratch');

% create SVM model
ind_mix = randperm(length(y_scratch));
X_scratch = X_scratch(ind_mix,:);
y_scratch = y_scratch(ind_mix,:);

X_scratch_train = X_scratch(1:1000,:);
X_scratch_test = X_scratch(1001:end,:);

y_scratch_train = y_scratch(1:1000,:);
y_scratch_test = y_scratch(1001:end,:);

%% train model linear

C = 0.1;
model = svmTrain(X_scratch_train, y_scratch_train, C, @linearKernel);

p_scratch_train = svmPredict(model, X_scratch_train);
p_scratch_test = svmPredict(model, X_scratch_test);

fprintf('Training Accuracy: %f\n', mean(double(p_scratch_train == y_scratch_train)) * 100);
fprintf('Testing Accuracy: %f\n', mean(double(p_scratch_test == y_scratch_test)) * 100);

%[C, sigma] = dataset3Params(X_scratch_train, y_scratch_train, X_scratch_test, y_scratch_test);

% train model with kernels

%find best C and sigma
%[C, sigma] = dataset3Params(X_scratch_train, y_scratch_train, X_scratch_test, y_scratch_test);


model = svmTrain(X_scratch_train, y_scratch_train, 0.1, @(x1, x2)gaussianKernel(x1, x2, 30));
p_scratch_train = svmPredict(model, X_scratch_train);
p_scratch_test = svmPredict(model, X_scratch_test);

fprintf('Training Accuracy: %f\n', mean(double(p_scratch_train == y_scratch_train)) * 100);
fprintf('Testing Accuracy: %f\n', mean(double(p_scratch_test == y_scratch_test)) * 100);