clc, clear, close all

dinfo = dir('./results/*.txt');
for K = 1 : length(dinfo)
  thisfilename = dinfo(K).name;  %just the name
  data = load(strcat('./results/', thisfilename)); %load just this file
  
  [filepath,name,ext] = fileparts(thisfilename)
  
  fileInfo = strsplit(name, '_')
  classifier = string(fileInfo(1))
  dataType = string(fileInfo(2))
  
  
  
  figure
  subplot(2, 1, 1)
  plot(data(:,1), data(:,2), 'b-o')
  str = sprintf('Training time for %s classifier on %s data', classifier, dataType);
  title(str)
  xlabel('training samples')
  ylabel('seconds')
  ylim([0 data(10,2)])
  xlim([0 data(10,1)])
  grid on
  
  subplot(2, 1, 2)
  plot(data(:,1), data(:,5), 'g-o', data(:,1), data(:,3) - data(:,5), 'r-o')
  ylim([0 100])
  xlim([0 data(10,1)])
  str = sprintf('Test performance for %s classifier on %s data', classifier, dataType);
  title(str)
  xlabel('training samples')
  ylabel('percent')
  legend('% correct', '% error', 'location', 'e')
  grid on
  
end