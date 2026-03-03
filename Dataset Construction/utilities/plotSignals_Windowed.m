function plotSignals_Windowed(fileList, time, data, numCh, windowLength)
% plotSignals_Windowed
%
% Description:
%   Plots multichannel time-domain signals from multiple files using a
%   fixed-length scrollable time window with an interactive slider.
%
% Inputs:
%   fileList     - Structure array containing file information (e.g., names)
%   time         - Cell array of time vectors corresponding to each file
%   data         - Cell array containing signal data for each file
%                  (channels × samples)
%   numCh        - Cell array specifying the number of channels per file
%   windowLength - Length of the displayed time window in seconds
%
% Outputs:
%   None. The function generates interactive figures for visualization.


    %windowLength = 10;  % in seconds

    for file = 1:length(fileList)
        t = time{file};       % time vector for this file
        fsDuration = t(end);  % total duration in seconds

        for ch = 1:numCh{file}
            % Create figure
            fig = figure('Name', ['Data from ', fileList(file).name], ...
                         'NumberTitle', 'off', ...
                         'Position', [100 100 2000 400]);

            % Create axes
            hAx = axes('Parent', fig);
            plot(hAx, t, data{file}(ch,:));
            grid(hAx, 'on');
            xlabel(hAx,'Time (s)');
            ylabel(hAx,'Amplitude');
            title(hAx, ['File: ' fileList(file).name ' | Channel ' num2str(ch)]);
            xlim(hAx, [0 windowLength]);  % initial window

            % Create slider
            sld = uicontrol('Style','slider', ...
                'Min', 0, ...
                'Max', fsDuration - windowLength, ...
                'Value', 0, ...
                'Units','normalized', ...
                'Position', [0.2 0.02 0.6 0.04], ...
                'SliderStep', [0.001 0.01], ...
                'Callback', @(src,~) updateWindowNormal(src.Value, hAx, windowLength));

            % add label to show current start time
            txt = uicontrol('Style','text', ...
                'Units','normalized', ...
                'Position',[0.82 0.02 0.15 0.04], ...
                'String', sprintf('Start: %.2f s', sld.Value));

            % Update text label whenever slider moves
            addlistener(sld, 'ContinuousValueChange', @(src,~) set(txt,'String',sprintf('Start: %.2f s', src.Value)));

            drawnow;
        end
    end
end

