input_path = "all";
output_path = "dev";
aug_factor = 3;

start = 1;
stop = -10;

% Get no of available data files
inputFiles = dir( fullfile(input_path, '*.mat') );
filenames = {inputFiles.name};
file_pattern = "fr";
iter = 1;
n_files = length(filenames);

for i = start:n_files+stop
  thisFileName = filenames{i};

  % Prepare the input filename.
  inputFullFileName = fullfile(pwd, input_path, thisFileName);

  % Prepare the output filename.
  for j = 1:aug_factor
      outputBaseFileName = strcat(file_pattern, num2str(iter), ".mat");
      outputFullFileName = fullfile(output_path, outputBaseFileName);
      iter = iter + 1;

      copyfile(inputFullFileName, outputFullFileName);
  end
end
